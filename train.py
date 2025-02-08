# Libraries
import gc
import json
import torch
import logging
import argparse
import configparser

# Local files
from Trainer import Trainer, TrainerConfig
from prepare_dataset import CharE, Tokenizer, CharDataset, remove_diacritics
from utils import seed_everything, check_dic, num_parameters
from model import Config, GPT



logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)
logger = logging.getLogger(__name__)


# Argument command line parser for config file
parser = argparse.ArgumentParser(prog="Train")
parser.add_argument("--config_path", help="Pass config file", metavar="FILE")
parser.add_argument("--model_config_section", help="Model section in config file", required=False, default="Model")
parser.add_argument("--train_batch_size", help="Train batch size", required=False, type=int, default=64)
parser.add_argument("--eval_batch_size", help="Eval batch size", required=False, type=int, default=64)    
parser.add_argument("--epoch", help="Number of training epoch", required=False, type=int, default=100)
parser.add_argument("--learning_rate", help="Training Learning rate", required=False, type=float, default=2e-4)
parser.add_argument("--seed", help="Set Randomised seed", required=False, type=int, default=42)   
parser.add_argument('--num_generated_tokens', help="Number of tokens to generate at inference", type=int, required=False, default=512)
parser.add_argument("--show_generated_text", help="Show the model generated text", required=False, default=True, type=bool)

args = parser.parse_args()

SEED = args.seed
seed_everything(SEED)

config_path = args.config_path
config = configparser.ConfigParser()
config.read(config_path)

#===============================[Model]===============================#
torch.set_float32_matmul_precision("high")

model_config_section = args.model_config_section

model_config = Config(config_file_path=config_path, section=model_config_section)
model = GPT(config=model_config)
model = torch.compile(model=model)

#===============================[Dataset]===============================#
dataset_path = config.get(section="dataset", option="dataset_path")

text = open(file=dataset_path, mode="r", encoding="utf-8").read()
text = remove_diacritics(text=text)
n = len(text)
train_size = int(n * 0.9)

char_encoding = CharE(text[: train_size])
char_encoding.form_token_map()

tokenizer = Tokenizer()
vocab_size = tokenizer.get_vocab_size()

train_data = text[: train_size]
val_data = text[train_size: ]

block_size = config.getint(section="Model", option="block_size")

train_dataset = CharDataset(train_data, block_size, tokenizer)
eval_dataset = CharDataset(val_data, block_size, tokenizer)


#===============================[Train Args]===============================#
learning_rate = args.learning_rate
max_epoch = args.epoch
train_batch_size = args.train_batch_size
eval_batch_size = args.eval_batch_size
num_generated_tokens = args.num_generated_tokens
show_generated_text = args.show_generated_text
num_workers = config.getint(section="training_config", option="num_workers")
ckpt_path = config.get(section="training_config", option="ckpt_path")
generatation_save_dict = config.get(section="training_config", option="generatation_save_dict")
save_file_name = config.get(section="training_config", option="save_file_name")

check_dic(dic_path=ckpt_path)
check_dic(dic_path=generatation_save_dict)

#===============================[Training Config]===============================#
training_config = TrainerConfig(
        max_epoch=max_epoch,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        learning_rate=learning_rate,
        ckpt_path=ckpt_path,
        seed=SEED,
        num_generated_tokens=num_generated_tokens,
        tokenizer=tokenizer,
        generatation_save_path=f"{generatation_save_dict}/{save_file_name}",
        show_generated_text=show_generated_text,
    )

#===============================[Run]===============================#
if __name__ == "__main__":
    trainer = Trainer(model, train_dataset, eval_dataset, training_config)
    trainer.train()

    # Write args to file
    args_file_name = f"{ckpt_path}/train_args.txt"
    
    with open(args_file_name, "w") as arg_file:
        json.dump(args.__dict__, arg_file, indent=2)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()