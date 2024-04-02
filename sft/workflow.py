# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/summarization/run_summarization.py

from typing import TYPE_CHECKING, List, Optional

from transformers import TrainingArguments

from ...data import get_dataset, split_dataset
from ...extras.constants import IGNORE_INDEX
from ...extras.misc import get_logits_processor
from ...extras.ploting import plot_loss
from ...model import load_model_and_tokenizer
from ...train.sft.metric import ComputeMetrics
from ...train.sft.tatqa_trainer import TATTrainer
from ...train.utils import create_modelcard_and_push
from custom_batch_sampler import custom_data, CustomBatchSampler,eval_data,EvalBatchSampler
import os
import pickle
import torch
import random
import numpy as np
from torch.utils.data import Dataset,BatchSampler,dataloader,RandomSampler


if TYPE_CHECKING:
    from transformers import TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "TrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    model = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train)
    # dataset = get_dataset(tokenizer, model_args, data_args, training_args, stage="sft")

    dataset = custom_data(data_args.cache_path+'train.pkl')

    sampl = CustomBatchSampler(RandomSampler(dataset),batch_size=training_args.train_batch_size, drop_last=False)

    sampl.get_num_ops(6)
    train_dataloader =  dataloader.DataLoader(
                dataset=dataset,
                batch_sampler=sampl,
                num_workers = 2,
            )


    eval_dataset = eval_data(data_args.cache_path+'eval.pkl')

    eval_sampl = EvalBatchSampler(RandomSampler(eval_dataset),batch_size=8, drop_last=False)

    eval_sampl.get_num_ops(6)
    eval_dataloader =  dataloader.DataLoader(
                dataset=eval_dataset,
                batch_sampler=eval_sampl,
                num_workers = 0,
                #prefetch_factors = 1,
            )


    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    # Override the decoding parameters of Seq2SeqTrainer
    #training_args_dict = training_args.to_dict()
    #training_args = TrainingArguments(**training_args_dict)

    # Initialize our Trainer
    trainer = TATTrainer(
        callbacks = callbacks,
        model=model,
        args=training_args,
        train_dataloader = train_dataloader,
        eval_dataloader = eval_dataloader
        )

    # Keyword arguments for `model.generate`

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        #trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        if training_args.predict_with_generate:  # eval_loss will be wrong if predict_with_generate is enabled
            metrics.pop("eval_loss", None)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(dataset, metric_key_prefix="predict", **gen_kwargs)
        if training_args.predict_with_generate:  # predict_loss will be wrong if predict_with_generate is enabled
            predict_results.metrics.pop("predict_loss", None)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(predict_results)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
