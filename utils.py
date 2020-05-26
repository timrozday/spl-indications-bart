import os
import random
import json
import csv
import numpy as np
import pandas as pd
from collections import namedtuple
from tqdm.auto import tqdm
from pathlib import Path
from datetime import datetime
import itertools as it

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BartForConditionalGeneration, BartTokenizer, AutoModelWithLMHead, AutoConfig
import pytorch_lightning as pl

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import warnings
warnings.filterwarnings('ignore')

import pdb

class SummarizationDataset(Dataset):
    def __init__(self, tokenizer, data_path="./train.json", type_path="train", doc_size=1024, answer_size=128, targets_lim=6):
        super(SummarizationDataset,).__init__()

        self.tokenizer = tokenizer
        self.doc_size = doc_size
        self.answer_size = answer_size
        
        self.data = []
        with open(data_path,'rt') as f:
            json_data = json.load(f)
            for set_id,source,targets in tqdm(json_data, desc=f"Loading {type_path} data"):
                if len(source)==0:
                    continue

                # delete duplicates and tokenize
                targets = list({t.lower() for t in targets})
                targets_t = [self.tokenizer.encode(t, pad_to_max_length=False, return_tensors='pt', add_special_tokens=False, add_prefix_space=True)[0] for t in targets]
                
                if len(targets)>targets_lim:
                    continue
                if sum([len(t) for t in targets_t]) + len(targets_t) + 1 > self.answer_size:
                    continue
                    
#                 targets_str = ' [SEP] '.join(targets)
#                 targets_t = self.tokenizer.tokenize(targets_str, max_length=self.answer_size, pad_to_max_length=True).squeeze()

#                 targets_index = torch.zeros([targets_lim,2], dtype=torch.int64)
#                 i = 0
#                 for j in range(targets_t.shape[0]):
#                     if targets_t[j] == 0: continue
#                     if targets_t[j] == 2:
#                         targets_index[i][1] = j-1
#                         break
#                     if targets_t[j] == 50118:
#                         targets_index[i][1] = j-1
#                         i += 1
#                         continue
#                     if targets_index[i][0] == 0:
#                         targets_index[i][0] = j
#                     if targets_t[j] == 1:
#                         raise Exception("Undefined token (1) reached before end-tokens (2)")
                
                source_t = self.tokenizer.encode_plus(source, max_length=self.doc_size, pad_to_max_length=True, return_tensors='pt', return_attention_mask=True, add_prefix_space=True)
            
                # remove data that is too large
#                 if source_t['input_ids'][0][-1] > 1: continue
#                 if targets_t[-1] > 1: continue
            
                self.data.append([source, source_t['input_ids'].squeeze(), source_t['attention_mask'].squeeze(), targets, targets_t])
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        source, source_t, source_attention_mask, targets, targets_t = self.data[index]
        
        # randomly sort targets
        joined_targets_t = torch.ones(self.answer_size, dtype=torch.int64)
        joined_targets_t[0] = 0  # <s>
        pos = 0
        for t in random.sample(targets_t,len(targets_t)):
            pos += 1
            joined_targets_t[pos:pos+len(t)] = t
            pos += len(t)
            joined_targets_t[pos] = 50263  # 50263 = '<sep>', 2 = '</s>'
        joined_targets_t[pos] = 2  # </s>
        
        target_attention_mask = torch.ones(self.answer_size, dtype=torch.int64)
        target_attention_mask[joined_targets_t==1] = 0
        
        return {"source_ids": source_t,
                "source_mask": source_attention_mask,
                "target_ids": joined_targets_t,
                "target_mask": target_attention_mask}
    
    
class BartSystem(pl.LightningModule):

    def __init__(self, hparams, user_tokens=['<newline>', '<bullet>', '<sep>']):
        super(BartSystem, self).__init__()
        self.hparams = hparams
        self.hparams.model_type = self.hparams.model_type.lower()
        tokenizer = BartTokenizer.from_pretrained(
            self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
            do_lower_case=self.hparams.do_lower_case,
            cache_dir=self.hparams.cache_dir if self.hparams.cache_dir else None,
        )
            
        config = AutoConfig.from_pretrained(
            self.hparams.config_name if self.hparams.config_name else self.hparams.model_name_or_path,
            cache_dir=self.hparams.cache_dir if self.hparams.cache_dir else None,
            output_past = self.hparams.do_test,
            vocab_size=len(tokenizer)
        )

        model = BartForConditionalGeneration.from_pretrained(
            self.hparams.model_name_or_path,
            from_tf=bool(".ckpt" in self.hparams.model_name_or_path),
            config=config,
            cache_dir=self.hparams.cache_dir if self.hparams.cache_dir else None,
        )
        
        self.config, self.tokenizer, self.model = config, tokenizer, model
        self.loss = []  # for keeping track of average loss
        self.metrics = {}
        
        self.vocab = {v:k for k,v in self.tokenizer.get_vocab().items()}
    
    def is_logger(self):
        return self.trainer.proc_rank <= 0

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        if self.trainer.use_tpu:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        try: avg_loss =  sum(self.loss)/len(self.loss)
        except: avg_loss = -1
        self.loss = []
#         tqdm_dict = {"loss": "{:.3g}".format(avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        tqdm_dict = {"loss": "{:.3g}".format(avg_loss)}
        return tqdm_dict

    def _feature_file(self, mode):
        return os.path.join(
            self.hparams.data_dir,
            "cached_{}_{}_{}".format(
                mode,
                list(filter(None, self.hparams.model_name_or_path.split("/"))).pop(),
                str(self.hparams.doc_max_seq_length),
            ),
        )
        
    def forward(
        self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            lm_labels=lm_labels,
        )

#     def join_target_tokens(self, targets_t, length=128, end=False):
#         device = targets_t[0].device
#         result = torch.zeros([1,length], dtype=torch.int64).to(device)
#         pos = 1
#         for t in targets_t:
#             result[0,pos:pos+len(t[0][1:-1])] = t[0][1:-1]
#             pos += len(t[0][1:])
#             result[0,pos-1] = 50118.0
#         if end:
#             result[0,pos-1] = 2.0
#         else:
#             result[0,pos-1] = 0.0

#         return result

    def _step(self, batch):  # delim_token=50118, start_token=0, end_token=2, undefined_token=1
        input_ids = batch["source_ids"].clone()
        
        attention_mask = batch["source_mask"].clone()
        
        decoder_input_ids = batch["target_ids"][:,:-1].clone()
        
        decoder_attention_mask = batch["target_mask"].clone()
        
        lm_labels = batch["target_ids"][:, 1:].clone()
        lm_labels[lm_labels == self.tokenizer.pad_token_id] = -100
        
        output = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
#             decoder_attention_mask=decoder_attention_mask
            lm_labels=lm_labels,
        )
        
        return output[0]
    
#         device = batch['target_ids'].device
        
#         n_targets = batch['targets_index'].shape[1]  # number of targets
#         len_targets = batch['target_ids'].shape  # length of targets tensore

#         target_index = batch['targets_index'][0]

#         y = torch.ones(len_targets, dtype=torch.int64, device=device)*undefined_token  # the final output targets tensor
#         target_order = torch.ones(n_targets, dtype=torch.int64, device=device)*-1  # used to prevent target being added to the order more than once
#         losses = torch.zeros(n_targets, device=device)  # used to store losses so that the smallest can be picked

#         y[0][0] = start_token  # set first token
#         pos = 0  # pos holds the posision to write to in y

#         for i in range(n_targets):  # in each iteration, add a target to y

#             # get loss of each of the potential new targets
#             for j in range(n_targets):

#                 # check if there is a target by checking the target_index
#                 if target_index[j][1] <= 0:
#                     losses[j] = -1
#                     continue

#                 # check that target is not already in target order
#                 same = False
#                 for k in range(n_targets):
#                     if j == target_order[k]:
#                         same = True
#                         break
#                 if same:
#                     losses[j] = -1
#                     continue

#                 # add this target to ordered_targets_map
#                 temp_y = y.clone()
#                 temp_pos = pos
#                 temp_y[0][temp_pos] = delim_token
#                 temp_pos += 1
#                 for k in range(target_index[j][0], target_index[j][1]+1):
#                     temp_y[0][temp_pos] = batch['target_ids'][0][k]
#                     temp_pos += 1
#                 temp_y[0][temp_pos] = end_token  # don't update pos, this is so that the end token gets overwritten when a new target is added
#                 temp_y[0][0] = start_token

#                 # get the loss
#                 y_ids = temp_y[:,:-1].contiguous()
#                 lm_labels = temp_y[:,1:].clone()
#                 lm_labels[lm_labels == self.tokenizer.pad_token_id] = -100

#                 output = self.model.forward(
#                     input_ids=batch["source_ids"],
#                     attention_mask=batch["source_mask"],
#                     decoder_input_ids=y_ids,
#                     lm_labels=lm_labels,
#                 )

#                 losses[j] = output[0].item()

#                 change = True

#             # find the target with the lowest loss
#             min_loss_i = -1
#             for j in range(n_targets):
#                 if losses[j] >= 0:
#                     if min_loss_i < 0:
#                         min_loss_i = j
#                     elif losses[j] < losses[min_loss_i]:
#                         min_loss_i = j

#             if min_loss_i >= 0:  # check if there is a target to process
#                 # add to y and target_order
#                 target_order[i] = min_loss_i
#                 y[0][pos] = delim_token
#                 pos += 1
#                 for k in range(target_index[min_loss_i][0], target_index[min_loss_i][1]+1):
#                     y[0][pos] = batch['target_ids'][0][k]
#                     pos += 1
#                 y[0][pos] = end_token  # don't update pos, this is so that the end token gets overwritten when a new target is added
#                 y[0][0] = start_token
#             else:
#                 break  # if there are no more targets to process, just exit the loop
# 
#         return output[0]  # i.e. loss of final y, which is the optimal combination of targets 
                    
        
#         n = len(batch['target_ids'])
#         ordered_targets = []
#         losses = {}
#         while len(ordered_targets)<n:
#             for target_i in set(range(n))-set(ordered_targets):
#                 y = self.join_target_tokens(
#                     [batch['target_ids'][i] for i in ordered_targets+[target_i]], 
#                     length=self.hparams.answer_max_seq_length, 
#                     end=(len(ordered_targets)==n)
#                 )
#                 y_ids = y[:,:-1].contiguous()
#                 lm_labels = y[:,1:].clone()
#                 lm_labels[y[:,1:] == self.tokenizer.pad_token_id] = -100

#                 input_ids = batch["source_ids"]
#                 attention_mask = batch["source_mask"]
                
#                 output = self.model.forward(
#                     input_ids=input_ids,
#                     attention_mask=attention_mask,
#                     decoder_input_ids=y_ids,
#                     lm_labels=lm_labels,
#                 )
                
#                 loss = output[0].clone()
#                 losses[loss.item()] = target_i
                
#             min_loss_i = losses[min(losses.keys())]
#             ordered_targets.append(min_loss_i)
        
#         y = batch["target_ids"]
#         y_ids = y[:, :-1].contiguous()
#         lm_labels = y[:, 1:].clone()
#         lm_labels[y[:, 1:] == self.tokenizer.pad_token_id] = -100
#         outputs = self(
#             input_ids=batch["source_ids"],
#             attention_mask=batch["source_mask"],
#             decoder_input_ids=y_ids,
#             lm_labels=lm_labels,
#         )
        
#         loss = outputs[0]
        
#         return loss

    def training_step(self, batch, batch_idx):
#         targets = self.train_dataset[batch_idx]['targets']

        loss = self._step(batch)
        self.loss.append(loss.item())  # for keeping track of average loss

#         tensorboard_logs = {"Training/Loss": loss}
        return {"loss": loss}  # "log": tensorboard_logs

    def validation_step(self, batch, batch_idx):
#         targets = self.val_dataset[batch_idx]['targets']

        loss = self._step(batch)   
        self.loss.append(loss.item())  # for keeping track of average loss

        return {"val_loss": loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
#         tensorboard_logs = {"Validation/Loss": avg_loss}
        return {"avg_val_loss": avg_loss}  # "log": tensorboard_logs

    def test_step(self, batch, batch_idx):
        generated_ids = self.model.generate(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_start_token_id=self.model.config.eos_token_id,
            
            max_length=self.hparams.answer_max_seq_length + 2,
            min_length=0,
            early_stopping=True,
            
            num_beams=4,
            do_sample=True,
            num_return_sequences=1,
            repetition_penalty=1.0,
            length_penalty=1.0,
        )

        preds = [
            self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for g in generated_ids
        ]
        target = [
            self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for t in batch["target_ids"]
        ]
        loss = self._step(batch)

        return {"val_loss": loss.item(), "preds": preds[0], "target": target[0]}

    def test_end(self, outputs):
        hits = 0
        total_targets = 0
        total_outputs = 0
        
        for output in outputs:
            val_targets = {s.strip().lower() for s in output["target"].split('\n')}
            gen_outputs = {s.strip().lower() for s in output['preds'].split('\n')}
            hits += len(val_targets & gen_outputs)
            total_targets += len(val_targets)
            total_outputs += len(gen_outputs)
                
        precision = hits/total_outputs
        recall = hits/total_targets
        f1 = 2*precision*recall/(precision+recall)
        
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
#         tensorboard_logs = {"Test/Loss": avg_loss, "Test/Precision": precision, "Test/Recall": recall, "Test/F1": f1}
        
        return {"avg_val_loss": avg_loss.item(), "precision": precision, "recall": recall, "f1": f1}  # "log": tensorboard_logs

    def test_epoch_end(self, outputs):
        output_test_predictions_file = os.path.join(self.hparams.output_dir, "test_predictions.txt")
        output_test_targets_file = os.path.join(self.hparams.output_dir, "test_targets.txt")
        output_test_metrics_file = os.path.join(self.hparams.output_dir, "test_metrics.txt")
        # write predictions and targets for later rouge evaluation.
        with open(output_test_predictions_file, "w+") as p_writer, open(output_test_targets_file, "w+") as t_writer:
            for output_batch in outputs:
                p_writer.write(output_batch["preds"]+"\n")
                t_writer.write(output_batch["target"]+"\n")
            p_writer.close()
            t_writer.close()
        
        # setup file if it's empty
        if not os.path.exists(output_test_metrics_file):
            with open(output_test_metrics_file, "w+") as m_writer:
                columns = ["precision", "recall", "f1"]
                output_file.write(",".join(columns)+"\n")
                
        r = self.test_end(outputs)
        with open(output_test_metrics_file, "w+") as m_writer:
            m_writer.write(",".join([f"{r[k]:.3g}" for k in ["precision","recall","f1"]])+"\n")
            m_writer.close()
            
        return r

    def train_dataloader(self):
        self.train_dataset = SummarizationDataset(
            self.tokenizer, data_path=self.hparams.train_data_path, type_path="train", doc_size=self.hparams.doc_max_seq_length, \
            answer_size=self.hparams.answer_max_seq_length,
        )
        dataloader = DataLoader(self.train_dataset, batch_size=self.hparams.train_batch_size, shuffle=self.hparams.shuffle_training_data, drop_last=True)

#         t_total = (
#             (len(dataloader.dataset) * float(self.hparams.num_train_epochs))
#             // (self.hparams.train_batch_size * self.hparams.gradient_accumulation_steps)
#         )
#         warmup_steps = self.hparams.warmup_steps*self.hparams.train_batch_size
#         scheduler = get_linear_schedule_with_warmup(
#             self.opt, num_warmup_steps=warmup_steps, num_training_steps=t_total
#         )

        schedule_freq = self.hparams.schedule_every_n_steps // (self.hparams.gradient_accumulation_steps * self.hparams.train_batch_size)
    
        scheduler = torch.optim.lr_scheduler.StepLR(
            self.opt,
            step_size=schedule_freq,
            gamma=self.hparams.lr_decay
        )

#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             self.opt,
#             mode='min',
#             factor=0.1,
#             patience=2,
#             verbose=True,
#             threshold=1e-4,
#             threshold_mode='rel',
#             cooldown=0,
#             min_lr=0,
#             eps=1e-08
#         )

        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        self.val_dataset = SummarizationDataset(
            self.tokenizer, data_path=self.hparams.val_data_path, type_path="val", doc_size=self.hparams.doc_max_seq_length, \
            answer_size=self.hparams.answer_max_seq_length,
        )
        return DataLoader(self.val_dataset, batch_size=self.hparams.eval_batch_size, shuffle=False, drop_last=True)

    def test_dataloader(self):
        self.test_dataset = SummarizationDataset(
            self.tokenizer, data_path=self.hparams.test_data_path, type_path="test", doc_size=self.hparams.doc_max_seq_length, \
            answer_size=self.hparams.answer_max_seq_length,
        )
        return DataLoader(self.test_dataset, batch_size=self.hparams.test_batch_size, shuffle=False, drop_last=True)


class TensorBoardCallback(pl.Callback):
    writer = None
    step_count = 0
    
    def on_test_start(self, trainer, pl_module):
        if not self.writer:
            self.writer = SummaryWriter(log_dir=pl_module.hparams.log_dir, max_queue=10, flush_secs=120)
        
    def on_sanity_check_start(self, trainer, pl_module):
        self.writer = SummaryWriter(log_dir=pl_module.hparams.log_dir, max_queue=10, flush_secs=120)
        self.weights_keys = [f'encoder.layer.{i}.output.LayerNorm.weight' for i in range(24)]
        self.biases_keys = [f'encoder.layer.{i}.output.LayerNorm.bias' for i in range(24)]
        self.val_losses = []
        self.step_count = 0
    
    def on_batch_end(self, trainer, pl_module):
        for i in range(pl_module.hparams.train_batch_size):
            self.step_count += 1
            if self.step_count % pl_module.hparams.hist_log_freq == 0:
                self.writer.add_scalar("Training/Learning-rate", pl_module.lr_scheduler.get_last_lr()[-1], self.step_count)  # log learning rate
                # log gradients, not sure how to do this...

                # log weights and biases
#                 state = dict(pl_module.model.bert.state_dict())
#                 for i,k in enumerate(self.weights_keys):
#                     v = state[k]
#                     title = f'Weights/Layer-{i}'
#                     self.writer.add_histogram(title, v, self.step_count)
#                 for i,k in enumerate(self.biases_keys):
#                     v = state[k]
#                     title = f'Biases/Layer-{i}'
#                     self.writer.add_histogram(title, v, self.step_count)
                    
        loss = trainer.running_loss.last()  # log loss
        if loss:
            self.writer.add_scalar("Training/Loss", loss.item(), self.step_count)  # log training loss
        
#         self.writer.flush()
        
    def on_validation_start(self, trainer, pl_module):
        self.val_losses = []
        
    def on_validation_batch_end(self, trainer, pl_module):
        loss = trainer.running_loss.last()
        if loss:
            self.val_losses.append(loss.item())
        
    def on_validation_end(self, trainer, pl_module):
        if len(self.val_losses):
            avg_loss = sum(self.val_losses)/len(self.val_losses)
            self.writer.add_scalar("Validation/Loss", avg_loss, self.step_count)  # log validation loss
#             self.writer.flush()

    def on_test_end(self, trainer, pl_module):
        for metric_type, metric_data in pl_module.metrics.items():
            for metric_name, value in metric_data.items():
                self.writer.add_scalar(f"Test/{metric_type}-{metric_name}", value, self.step_count)  # log validation loss
        #  self.writer.flush()

class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            # Log and save results to file
            output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info(f"{key} = {metrics[key]:.3g}\n")
                        writer.write(f"{key} = {metrics[key]:.3g}\n")

class EpochCallback(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        # checkpoint
        Path(pl_module.hparams.output_dir).mkdir(parents=True, exist_ok=True)
        fn = f"{pl_module.hparams.output_dir}/checkpoint-step-{trainer.global_step}.ckpt"
        trainer.save_checkpoint(fn)

        fp = f"{pl_module.hparams.output_dir}/checkpoint-{trainer.global_step}"
        Path(fp).mkdir(parents=True, exist_ok=True)
        pl_module.model.save_pretrained(fp)
        pl_module.tokenizer.save_pretrained(fp)
        pl_module.config.save_pretrained(fp)
            
        if pl_module.is_logger():
            logger.info(f"Model checkpointed at step {trainer.global_step}")
                        
class CheckpointCallback(pl.Callback):
    def on_batch_end(self, trainer, pl_module):
        if trainer.global_step % (pl_module.hparams.checkpoint_every_n_steps//pl_module.hparams.train_batch_size) == 0:
            # checkpoint
            Path(pl_module.hparams.output_dir).mkdir(parents=True, exist_ok=True)
            fn = f"{pl_module.hparams.output_dir}/checkpoint-step-{trainer.global_step}.ckpt"
            trainer.save_checkpoint(fn)
            
            fp = f"{pl_module.hparams.output_dir}/checkpoint-{trainer.global_step}"
            Path(fp).mkdir(parents=True, exist_ok=True)
            pl_module.model.save_pretrained(fp)
            pl_module.tokenizer.save_pretrained(fp)
            pl_module.config.save_pretrained(fp)
            
            if pl_module.is_logger():
                logger.info(f"Model checkpointed at step {trainer.global_step}")
        
class DebugCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        from gpu_profile import gpu_profile
        import sys
        sys.settrace(gpu_profile)

        
        
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class Dict2Obj(dict):
    """
    Example:
    m = Dict2Obj({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super(Dict2Obj, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        try: return self.__dict__[attr]
        except KeyError: raise AttributeError(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Dict2Obj, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Dict2Obj, self).__delitem__(key)
        del self.__dict__[key]
        

def get_trainer(pl_module, args):
    # init model
    set_seed(args)

    t = datetime.now()
    prefix = t.strftime('%d%b%Y-%H:%M:%S')

    # Create output dir
    if args.do_train:  #  os.path.exists(args.output_dir) and os.listdir(args.output_dir) and
        args.output_dir = os.path.join(args.output_dir, f"{prefix}")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create log dir
    args.log_dir = os.path.join(args.log_dir, f"{prefix}")
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    args.gpus = list(range(args.n_gpu))

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        auto_select_gpus=False,  # True might cause memory to run out
        max_epochs=args.num_train_epochs,
        early_stop_callback=False,
        gradient_clip_val=args.max_grad_norm,
#         checkpoint_callback=checkpoint_callback,
        callbacks=[LoggingCallback(), EpochCallback(), TensorBoardCallback()],
        reload_dataloaders_every_epoch=False,
        train_percent_check=args.train_percent_check,
        val_percent_check=args.val_percent_check,
        test_percent_check=args.test_percent_check,
    )

    if hasattr(args, 'profile_gpu_memory'):
        if args.profile_gpu_memory:
            train_params['callbacks'].append(DebugCallback())
    if hasattr(args, 'checkpoint_every_n_steps'):
        if args.checkpoint_every_n_steps > 0:
            train_params['callbacks'].append(CheckpointCallback())
    if hasattr(args, 'val_check_interval'):
        if args.val_check_interval > 0:
            train_params['val_check_interval'] = args.val_check_interval

    if args.fp16:
        train_params["use_amp"] = args.fp16
        train_params["amp_level"] = args.fp16_opt_level

    if args.n_gpu > 1:
        train_params["distributed_backend"] = "dp"
    else:
        train_params["distributed_backend"] = None

    trainer = pl.Trainer(**train_params)
    
    return trainer
        
def generic_train(trainer, pl_module, args):
    if args.do_train:
        trainer.fit(pl_module)

    return trainer
