# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.
import traceback

import base64
from io import BytesIO
import re
import contextlib
import os

import PIL
from PIL import ImageFile
from torchvision import transforms

from .transforms import *
from .input_dataset import FileDataset

from .collate_rec import (
    collate_fn,
)

import os,json

label_map = {"entailment": 0, "not_entailment": 1}

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

FLAMINGO_MEAN = [0.48145466, 0.4578275, 0.40821073]
FLAMINGO_STD = [0.26862954, 0.26130258, 0.27577711]

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

from torch.utils.data import Dataset

class RecDataset(Dataset):
    def __init__(self, args, is_test=False, supported_data_types=["seq_rec"], split="train", task="rec"):
        # super().__init__()
        self.split = split
        self.args = args
        self.task_name = task
        self.is_test = is_test
        self.tokenizer = args.tokenizer
        self.tasks = task
        self.use_semantic=args.use_semantic
        self.max_src_length = args.max_src_length
        self.max_tgt_length = args.max_tgt_length

        self.seed = args.pretrain_seed
        self.code_dict_size = args.code_dict_size
        self.patch_image_size = args.patch_image_size
        self.code_image_size = args.code_image_size
        self.supported_data_types = supported_data_types

        self.epoch = 0

        scales = [(args.patch_image_size, args.patch_image_size)]
        

        # TODO: check if random augment is correct, especially for some questions related to colors.
        # self.patch_resize_transform = transforms.Compose(
        #     [
        #         RandomResize(scales),
        #         transforms.CenterCrop(args.patch_image_size),
        #         transforms.RandomHorizontalFlip(p=0.2),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=FLAMINGO_MEAN, std=FLAMINGO_STD),
        #     ]
        # )
        if split=="train":
            self.patch_resize_transform = transforms.Compose(
                [
                    RandomResize(scales),
                    # transforms.RandomHorizontalFlip(p=0.5),
                    # transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=FLAMINGO_MEAN, std=FLAMINGO_STD),
                ]
            )
        else:
            self.patch_resize_transform = transforms.Compose(
                [
                    RandomResize(scales),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=FLAMINGO_MEAN, std=FLAMINGO_STD),
                ]
            )
        self.folder_path = args.mmrec_path
        self.subset = args.subset
        
        self.img_folder = os.path.join(self.folder_path, self.subset)
        # ！！！
        # self.data_path = os.path.join(self.folder_path, f"{split}_{self.subset}.json")
        self.data_path = os.path.join(self.folder_path, f"{split}_users.json")
        # ！！！
        self.data_meta2id_path = os.path.join(self.folder_path, f"{split}_{self.subset}_meta2id.json")
        self.data_id2meta_path = os.path.join(self.folder_path, f"{split}_{self.subset}_id2meta.json")
        self.data_img_gen_path = os.path.join(self.folder_path, f"{split}_{self.subset}.json")
        self.retrieval_data_path = os.path.join(self.folder_path, f"search_merge_{split}.txt")
        self.data_search_path = self.data_path
        self.meta_path = os.path.join(self.folder_path, f"meta_{self.subset}.json")
        # img semantic id
        self.img_id_path = os.path.join(self.folder_path, f"img_id2semantic.json")
        

        with open(self.data_path) as f:
            self.data = json.load(f)

        print(f"from {self.data_path} load {len(self.data)} number of samples")


        # semantic id
        if args.use_semantic:
            self.len_semanticid = 3
            self.id_path = os.path.join(self.folder_path, f"id2semantic.json")
            with open(self.id_path) as f:
                self.id2semantic = json.load(f)
            
            
        # history length
        if self.subset=="all":
            if self.tasks=="img_gen":
                self.history_len = 2
            else:
                self.history_len = 5
        if self.subset=="netflix":
            self.history_len = 3
        if self.subset=="hm":
            self.history_len = 8
        with open(self.meta_path) as f:
            self.meta_data = json.load(f)
        if split=="train":
            if args.single_task:
                if self.tasks=="rec":
                    with open(self.data_path) as f:
                        self.data = json.load(f)
                    self.tasks = ["rec"]*len(self.data)
                    self.seqs = list(self.data.values())
                elif self.tasks=="id2meta":
                    with open(self.data_path) as f:
                        self.data = json.load(f)
                    self.tasks = ["id2meta"]*len(self.data)
                    self.seqs = list(self.data.values())
                elif self.tasks=="meta2id":
                    with open(self.data_path) as f:
                        self.data = json.load(f)
                    self.tasks = ["meta2id"]*len(self.data)
                    self.seqs = list(self.data.values())
                elif self.tasks=="search":
                    with open(self.data_search_path) as f:
                        self.data = json.load(f)
                    self.tasks = ["search"]*len(self.data)
                    self.seqs = list(self.data.values())
                elif self.tasks=="img_gen":
                    with open(self.retrieval_data_path) as f:
                        self.retrieval_data = json.load(f)
                    with open(self.img_id_path) as f:
                        self.img_id2semantic = json.load(f)
                    # pretrain!!!
                    # with open(self.meta_path) as f:
                    #     self.data = json.load(f)
                    with open(self.data_img_gen_path) as f:
                        self.data = json.load(f)
                    self.tasks = ["img_gen"]*len(self.data)
                    self.seqs = self.retrieval_data
                self.keys = list(self.data.keys())
            elif type(task)==list:
                self.t2id={"id2meta":0, "meta2id":1, "rec":2}
                with open(self.data_path) as f:
                    self.rec_data = json.load(f)
                with open(self.data_path) as f:
                    self.id2meta_data = json.load(f)
                    print("加载成功2")
                with open(self.data_path) as f:
                    self.meta2id_data = json.load(f)
                
                # with open(self.data_id2meta_path) as f:
                #     self.id2meta_data = json.load(f)
                # with open(self.data_meta2id_path) as f:
                #     self.sel_data = json.load(f)
                # with open(self.data_search_path) as f:
                #     self.search_data = json.load(f)
                self.data = [self.rec_data, self.id2meta_data, self.meta2id_data]
                self.seqs, self.tasks = [], []
                n = len(task)
                for i, t in enumerate(task):
                    ind = self.t2id[t]
                    t_data = self.data[ind]
                    if i<n-1:
                        t_keys = list(t_data.keys())
                        np.random.shuffle(t_keys)
                        n_keys = int(0.25*len(t_keys))
                        cur_keys = t_keys[:n_keys]
                        cur_data = {key: t_data[key] for key in cur_keys}
                    else:
                        cur_data = t_data
                    cur_seq = list(cur_data.values())
                    self.seqs += cur_seq
                    self.tasks += [t]*len(cur_seq)
            else:
                with open(self.data_path) as f:
                    self.rec_data = json.load(f)
                with open(self.data_path) as f:
                    self.id2meta_data = json.load(f)
                with open(self.data_path) as f:
                    self.meta2id_data = json.load(f)
                # with open(self.data_id2meta_path) as f:
                #     self.id2meta_data = json.load(f)
                # with open(self.data_meta2id_path) as f:
                #     self.sel_data = json.load(f)
                # with open(self.data_search_path) as f:
                #     self.search_data = json.load(f)
                # with open(self.data_img_gen_path) as f:
                #     self.img_gen_data = json.load(f)
                self.rec_data = list(self.rec_data.values())
                self.id2meta_data = list(self.id2meta_data.values())
                self.meta2id_data = list(self.meta2id_data.values())
                # self.sel_data = list(self.sel_data.values())
                # self.search_data = list(self.search_data.values())
                # self.img_gen_data = list(self.img_gen_data.values())
                self.seqs = self.rec_data+self.id2meta_data+self.meta2id_data
                self.tasks = ["rec"]*len(self.rec_data)+["id2meta"]*len(self.id2meta_data)
                +["meta2id"]*len(self.meta2id_data)
                #print("调试b",set(self.tasks))
                # +["img_gen"]*len(self.img_gen_data)
                print(f"len(self.id2meta_data){len(self.id2meta_data)}self.tasks{self.tasks}")
            # with open(self.data_search_path) as f:
            #     self.search_data = json.load(f)
            # self.search_data = list(self.search_data.values())
            # self.seqs = self.search_data
            # self.tasks = ["search"]*len(self.search_data)
            
        else:
            #!!!
            self.data_path = os.path.join(self.folder_path, f"test_users.json")
            self.data_search_path = os.path.join(self.folder_path, f"test_users.json")
            #!!!
            if self.tasks=="rec":
                with open(self.data_path) as f:
                    self.data = json.load(f)
                self.seqs = list(self.data.values())
            elif self.tasks=="id2meta":
                with open(self.data_path) as f:
                        self.data = json.load(f)
                self.seqs = list(self.data.values())
            elif self.tasks=="meta2id":
                with open(self.data_path) as f:
                        self.data = json.load(f)
                self.seqs = list(self.data.values())

            self.tasks = [self.tasks]*len(self.seqs)
            #print("调试a",set(self.tasks))
            self.keys = list(self.data.keys())
        if self.subset=="all":
            self.all_items = set(range(22738))
        if self.subset=="netflix":
            self.all_items = set(range(1870))
        if self.subset=="hm":
            self.all_items = set(range(14901))
        # self.all_items = set(range(12094))
        
        # self.tasks = "rec"
        
        # self.selected_col_ids = [
        #         int(col_id) for col_id in args.selected_col_ids.split(",")
        #     ]
        # self.dtypes = [str for col_id in self.selected_col_ids]

        # self.dataset = dataset
        self.bos_item = torch.LongTensor([args.tokenizer.bos_token_id])
        self.eos_item = torch.LongTensor([args.tokenizer.eos_token_id])
        self.bos_mask = torch.LongTensor([1])
        self.eos_mask = torch.LongTensor([1])
        self.rank = args.rank


    def set_epoch(self, epoch, **unused):
        self.epoch = epoch
        
    
    def extract_meta(self, index):
        max_length = 20
        sample = self.meta_data.get(str(index)) 
        if sample is None:
            category = "Unknown"
            brand = "Unknown"
            title = "Unknown"
            price = "Unknown"
        else:
            category = "Unknown" if sample["category"] == "" else sample["category"]
            category = " ".join(category.split()[:max_length])
            brand = "Unknown" if sample["brand"] == "" else sample["brand"]
            brand = " ".join(brand.split()[:max_length])
            title = "Unknown" if sample["title"] == "" else sample["title"]
            title = " ".join(title.split()[:max_length])
            price = "Unknown" if sample["price"] == "" else sample["price"]
        
        text = f"Category {category} Price {price} Brand {brand} Title {title}"
        return text
    
    def extract_meta_gen(self, index):
        max_length = 20
        sample = self.meta_data[str(index)]
        p = np.random.random()
        # if p<0.3:
        #     category = "Unknown"
        # else:
        category = "Unknown" if sample["category"]=="" else sample["category"]
        category = " ".join(category.split()[:max_length])
        brand = "Unknown" if sample["brand"]=="" else sample["brand"]
        brand = " ".join(brand.split()[:max_length])
        title = "Unknown" if sample["title"]=="" else sample["title"]
        title = " ".join(title.split()[:max_length])
        # description = "Unknown" if sample["description"]=="" else sample["description"]
        img_id = self.img_id2semantic[str(index)]
        img_id = [f"img_{id}," for i, id in enumerate(img_id)]
        img_id = "".join(img_id)
        text = f"Title {title} ID {img_id}"
        return text
    
    def extract_meta_netflix(self, index):
        max_length = 20
        sample = self.meta_data[str(index)]
        p = np.random.random()
        # if p<0.3:
        #     category = "Unknown"
        # else:
        year = sample[0]
        title = sample[1]
        title = " ".join(title.split()[:max_length])
        text = f"Title {title} Release Date {year}"
        return text
    
    def extract_meta_hm(self, index):
        max_length = 20
        sample = self.meta_data[str(index)]
        p = np.random.random()
        # if p<0.3:
        #     category = "Unknown"
        # else:
        prod_name, appearance, color, section, describe =  sample[0], sample[1], sample[2], sample[3], sample[4]
        prod_name = " ".join(prod_name.split()[:max_length])
        appearance = " ".join(appearance.split()[:max_length])
        color = " ".join(color.split()[:max_length])
        section = " ".join(section.split()[:max_length])
        # print(prod_name, appearance, color, section, describe)  Description {describe}
        describe = " ".join(describe.split()[:max_length])
        # text = f"Name {prod_name} Appearance {appearance} Color {color} Section {section} Description {describe}"
        text = f"Name {prod_name} Appearance {appearance} Color {color} Section {section}"
        # text = f"Name {prod_name}"
        # text = f"Name {prod_name} Section {section}"
        return text

    def process_train_rec_pair(self, index):
        full_seq = self.seqs[index]
        seq = [item[0] for item in full_seq]
        img_seq = []
        input_seq = ""
        len_seq = len(seq)
        # start = -min(20, len_seq)
        # lower = min(start+5,-1)
        # end = np.random.choice(list(range(lower,0)),1)[0]
        
        # if len_seq<5:
        #     end = -1
        #     start = -len_seq
        # else:
        #     end = np.random.choice(list(range(-(len_seq-4),0)),1)[0]
        #     maxi = max(-len_seq, end-20)
        #     start = np.random.choice(list(range(maxi,end-3)),1)[0]
            
        # start=-20
        # end=-1
        
        # eqaul sequence length
        start = np.random.choice(list(range(0,len_seq-self.history_len)), 1)[0]
        end = start+self.history_len
        for item in seq[start:end]:
            image_item = Image.open(os.path.join(self.img_folder, f"{item}.jpg")).convert("RGB")
            img_seq.append(self.patch_resize_transform(image_item))
            # meta_item = self.extract_meta(item)
            if self.subset=="all":
                meta_item = self.extract_meta(item)
            elif self.subset=="netflix":
                meta_item = self.extract_meta_netflix(item)
            elif self.subset=="hm":
                meta_item = self.extract_meta_hm(item)
            # if not self.use_semantic:
            #     id_ = f"item_{item}"
            # else:
            #     id_ = self.id2semantic[str(item)].split(",")
            #     # semantic_id = [f"item_{i}_{id}" for i, id in enumerate(semantic_id)]
            #     id_ = [f"item_{id}" if i<self.len_semanticid else f"item_last_{id}" for i, id in enumerate(id_)]
            #     id_ = " ".join(id_)
            if not self.use_semantic:
                item_seq = f"<image> <answer> {meta_item} item_{item} <|endoftext|> "
                # item_seq = f"<image> {meta_item} <answer> item_domain_{item} <|endofchunk|> "
            else:
                semantic_id = self.id2semantic[str(item)].split(",")
                semantic_id = [f"item_{id}" if i<self.len_semanticid else f"item_last_{id}" for i, id in enumerate(semantic_id)]
                semantic_id = "".join(semantic_id)
                item_seq = f"<image> {meta_item} <answer> {semantic_id} <|endofchunk|> "
            input_seq = input_seq+item_seq
        if not self.use_semantic:
            # naive id
            input_seq = input_seq+f"<answer> item_{seq[end]} <|endofchunk|>"
            # input_seq = input_seq+f"What is the next item recommended to the user? <answer> item_domain_{seq[end]}"
        else:
            # semantic id
            item = seq[end]
            semantic_id = self.id2semantic[str(item)].split(",")
            # semantic_id = [f"item_{i}_{id}" for i, id in enumerate(semantic_id)]
            semantic_id = [f"item_{id}" if i<self.len_semanticid else f"item_last_{id}" for i, id in enumerate(semantic_id)]
            semantic_id = "".join(semantic_id)
            input_seq = input_seq+f"What is the next item recommended to the user? <answer> {semantic_id}"
        
        patch_image = torch.stack(img_seq,dim=0)
        src_text = self.tokenizer(
                input_seq,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True
            )
        src_item = src_text["input_ids"].squeeze(0)
        src_item_mask = src_text["attention_mask"].squeeze(0)
        # src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        # src_item_mask = torch.cat([self.bos_mask, src_item_mask, self.eos_mask])
        if src_item[-1] == 50280:
            src_item = torch.cat((src_item, torch.tensor([209])))
            src_item_mask = torch.cat((src_item_mask, torch.tensor([1])))
        elif src_item[-1] != 209:
            src_item = torch.cat((src_item, torch.tensor([209])))
            src_item_mask = torch.cat((src_item_mask, torch.tensor([1])))

        example = {
            "net_input":{
                "input_ids": src_item,
                "attention_masks": src_item_mask,
                "patch_images": patch_image,
                "weights": torch.tensor(2.0)
            }
        }

        return example

    def process_train_meta2id_pair(self, index):
        full_seq = self.seqs[index]
        seq = [item[0] for item in full_seq]
        img_seq = []
        input_seq = ""
        len_seq = len(seq)
        # start = -min(20, len_seq)
        # lower = min(start+5,-1)
        # end = np.random.choice(list(range(lower,0)),1)[0]
        
        # if len_seq<5:
        #     end = -1
        #     start = -len_seq
        # else:
        #     end = np.random.choice(list(range(-(len_seq-4),0)),1)[0]
        #     maxi = max(-len_seq, end-20)
        #     start = np.random.choice(list(range(maxi,end-3)),1)[0]
            
        # start=-20
        # end=-1
        
        # eqaul sequence length
        start = np.random.choice(list(range(0,len_seq-self.history_len)), 1)[0]
        end = start+self.history_len
        for item in seq[start:end]:
            image_item = Image.open(os.path.join(self.img_folder, f"{item}.jpg")).convert("RGB")
            img_seq.append(self.patch_resize_transform(image_item))
            # meta_item = self.extract_meta(item)
            if self.subset=="all":
                meta_item = self.extract_meta(item)
            elif self.subset=="netflix":
                meta_item = self.extract_meta_netflix(item)
            elif self.subset=="hm":
                meta_item = self.extract_meta_hm(item)
            # if not self.use_semantic:
            #     id_ = f"item_{item}"
            # else:
            #     id_ = self.id2semantic[str(item)].split(",")
            #     # semantic_id = [f"item_{i}_{id}" for i, id in enumerate(semantic_id)]
            #     id_ = [f"item_{id}" if i<self.len_semanticid else f"item_last_{id}" for i, id in enumerate(id_)]
            #     id_ = " ".join(id_)
            if not self.use_semantic:
                item_seq = f"{meta_item} <image> belongs to the description of an item, help me retrieve its item id? <answer> item_{item} <|endofchunk|> "
                # item_seq = f"<image> {meta_item} <answer> item_domain_{item} <|endofchunk|> "
            else:
                semantic_id = self.id2semantic[str(item)].split(",")
                semantic_id = [f"item_{id}" if i<self.len_semanticid else f"item_last_{id}" for i, id in enumerate(semantic_id)]
                semantic_id = "".join(semantic_id)
                item_seq = f"<image> {meta_item} <answer> {semantic_id} <|endofchunk|> "
            input_seq = input_seq+item_seq
        if not self.use_semantic:
            # naive id
            input_seq = input_seq+f"<answer> item_{seq[end]}"
            # input_seq = input_seq+f"What is the next item recommended to the user? <answer> item_domain_{seq[end]}"
        else:
            # semantic id
            item = seq[end]
            semantic_id = self.id2semantic[str(item)].split(",")
            # semantic_id = [f"item_{i}_{id}" for i, id in enumerate(semantic_id)]
            semantic_id = [f"item_{id}" if i<self.len_semanticid else f"item_last_{id}" for i, id in enumerate(semantic_id)]
            semantic_id = "".join(semantic_id)
            input_seq = input_seq+f"What is the next item recommended to the user? <answer> {semantic_id}"
        
        patch_image = torch.stack(img_seq,dim=0)
        src_text = self.tokenizer(
                input_seq,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True
            )
        src_item = src_text["input_ids"].squeeze(0)
        src_item_mask = src_text["attention_mask"].squeeze(0)
        # src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        # src_item_mask = torch.cat([self.bos_mask, src_item_mask, self.eos_mask])
        if src_item[-1] == 50280:
            src_item = torch.cat((src_item, torch.tensor([209])))
            src_item_mask = torch.cat((src_item_mask, torch.tensor([1])))
        elif src_item[-1] != 209:
            src_item = torch.cat((src_item, torch.tensor([209])))
            src_item_mask = torch.cat((src_item_mask, torch.tensor([1])))

        example = {
            "net_input":{
                "input_ids": src_item,
                "attention_masks": src_item_mask,
                "patch_images": patch_image,
                "weights": torch.tensor(2.0)
            }
        }

        return example

    def process_train_id2meta_pair(self, index):
        full_seq = self.seqs[index]
        seq = [item[0] for item in full_seq]
        img_seq = []
        input_seq = ""
        len_seq = len(seq)
        # start = -min(20, len_seq)
        # lower = min(start+5,-1)
        # end = np.random.choice(list(range(lower,0)),1)[0]
        
        # if len_seq<5:
        #     end = -1
        #     start = -len_seq
        # else:
        #     end = np.random.choice(list(range(-(len_seq-4),0)),1)[0]
        #     maxi = max(-len_seq, end-20)
        #     start = np.random.choice(list(range(maxi,end-3)),1)[0]
            
        # start=-20
        # end=-1
        
        # eqaul sequence length
        start = np.random.choice(list(range(0,len_seq-self.history_len)), 1)[0]
        end = start+self.history_len
        for item in seq[start:end]:
            image_item = Image.open(os.path.join(self.img_folder, f"{item}.jpg")).convert("RGB")
            img_seq.append(self.patch_resize_transform(image_item))
            # meta_item = self.extract_meta(item)
            if self.subset=="all":
                meta_item = self.extract_meta(item)
            elif self.subset=="netflix":
                meta_item = self.extract_meta_netflix(item)
            elif self.subset=="hm":
                meta_item = self.extract_meta_hm(item)
            # if not self.use_semantic:
            #     id_ = f"item_{item}"
            # else:
            #     id_ = self.id2semantic[str(item)].split(",")
            #     # semantic_id = [f"item_{i}_{id}" for i, id in enumerate(semantic_id)]
            #     id_ = [f"item_{id}" if i<self.len_semanticid else f"item_last_{id}" for i, id in enumerate(id_)]
            #     id_ = " ".join(id_)
            if not self.use_semantic:
                item_seq = f"item_{item} is the id of this item, please give me its detailed descriptions. <answer> <image> {meta_item} <|endofchunk|> "
                # item_seq = f"<image> {meta_item} <answer> item_domain_{item} <|endofchunk|> "
            else:
                semantic_id = self.id2semantic[str(item)].split(",")
                semantic_id = [f"item_{id}" if i<self.len_semanticid else f"item_last_{id}" for i, id in enumerate(semantic_id)]
                semantic_id = "".join(semantic_id)
                item_seq = f"<image> {meta_item} <answer> {semantic_id} <|endofchunk|> "
            input_seq = input_seq+item_seq
        if not self.use_semantic:
            # naive id
            input_seq = input_seq
            # input_seq = input_seq+f"What is the next item recommended to the user? <answer> item_domain_{seq[end]}"
        else:
            # semantic id
            item = seq[end]
            semantic_id = self.id2semantic[str(item)].split(",")
            # semantic_id = [f"item_{i}_{id}" for i, id in enumerate(semantic_id)]
            semantic_id = [f"item_{id}" if i<self.len_semanticid else f"item_last_{id}" for i, id in enumerate(semantic_id)]
            semantic_id = "".join(semantic_id)
            input_seq = input_seq+f"What is the next item recommended to the user? <answer> {semantic_id}"
        
        patch_image = torch.stack(img_seq,dim=0)
        src_text = self.tokenizer(
                input_seq,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True
            )
        src_item = src_text["input_ids"].squeeze(0)
        src_item_mask = src_text["attention_mask"].squeeze(0)
        # src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        # src_item_mask = torch.cat([self.bos_mask, src_item_mask, self.eos_mask])
        if src_item[-1] == 50280:
            src_item = torch.cat((src_item, torch.tensor([209])))
            src_item_mask = torch.cat((src_item_mask, torch.tensor([1])))
        elif src_item[-1] != 209:
            src_item = torch.cat((src_item, torch.tensor([209])))
            src_item_mask = torch.cat((src_item_mask, torch.tensor([1])))

        example = {
            "net_input":{
                "input_ids": src_item,
                "attention_masks": src_item_mask,
                "patch_images": patch_image,
                "weights": torch.tensor(2.0)
            }
        }

        return example
    

    
    def process_eval_rec_pair(self, index):
        #def process_eval_id2meta_pair(self, index): 替代
        full_seq = self.seqs[index]
        # print("开头seqs源数据",self.seqs)
        # print("开头full_seq完整内容",full_seq)
        #print("full_seq(1)",full_seq[1])
        seq = [item[0] for item in full_seq]
        img_seq = []
        input_seq = ""
        if self.subset=="hm":
            test_len = 20
        else:
            test_len = 5
        for item in seq[-test_len:-1]:
            # p = np.random.random()
            # if p<1.1:
            #     item_seq = f"{item} <|endofchunk|> "
            # else:
            # p = np.random.random()
            # if p<0.3:
            #     image_item = Image.open(os.path.join(self.img_folder, f"0.jpg")).convert("RGB")
            # else:
            image_item = Image.open(os.path.join(self.img_folder, f"{item}.jpg")).convert("RGB")
            
            # meta_item = self.extract_meta(item)
            if self.subset=="all":
                meta_item = self.extract_meta(item)
                meta_end = self.extract_meta(seq[-1])
            elif self.subset=="netflix":
                meta_item = self.extract_meta_netflix(item)
            elif self.subset=="hm":
                meta_item = self.extract_meta_hm(item)
            img_seq.append(self.patch_resize_transform(image_item))
            # semantic_id = f"item_{item}"
            # item_seq = f"<image> {meta_item} <|endofchunk|> "
            if not self.use_semantic:
                item_seq = f"Please tell me what item_{item} is called, along with a brief description of it. <answer> <|endofchunk|>"
                # item_seq = f"<image> {meta_item} item_domain_{item} <|endofchunk|> "
            else:
                semantic_id = self.id2semantic[str(item)].split(",")
                semantic_id = [f"item_{id}" if i<self.len_semanticid else f"item_last_{id}" for i, id in enumerate(semantic_id)]
                semantic_id = "".join(semantic_id)
                item_seq = f"<image> {meta_item} {semantic_id} <|endofchunk|> "
            input_seq = item_seq
        #input_seq = input_seq+f"<answer>"
        input_seq = input_seq
        #input_seq = input_seq+f"Based on the history that I provided , what is the next item recommended to the user? <answer>"
        
        # naive id
        if not self.use_semantic:
            #print("seq完整内容",seq)
            #print("full_seq完整内容",full_seq)
            semantic_id = f"item_{seq[-1]}"
            # semantic_id = f"item_domain_{seq[-1]}"
        else:
            # semantic id
            item = seq[-1]
            semantic_id = self.id2semantic[str(item)].split(",")
            # semantic_id = [f"item_{i}_{id}" for i, id in enumerate(semantic_id)]
            semantic_id = [f"item_{id}" if i<self.len_semanticid else f"item_last_{id}" for i, id in enumerate(semantic_id)]
            semantic_id = "".join(semantic_id)
        
        input_len = len(input_seq.split(" "))
        
        patch_image = torch.stack(img_seq,dim=0)

        src_text = self.tokenizer(
                input_seq,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True
            )
        
        #print("input_ids",input_ids)
        src_item = src_text["input_ids"].squeeze(0)
        src_item_mask = src_text["attention_mask"].squeeze(0)
        # print("input_ids",src_item)
        if src_item[-1] == 50280:
            src_item = torch.cat((src_item, torch.tensor([209])))
            src_item_mask = torch.cat((src_item_mask, torch.tensor([1])))
        elif src_item[-1] != 209:
            src_item = torch.cat((src_item, torch.tensor([209])))
            src_item_mask = torch.cat((src_item_mask, torch.tensor([1])))

        input_ids = src_text["input_ids"]
        #print("input_ids",input_ids)
        endofchunk_token_id = self.tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"][-1]
        answer_token_id = self.tokenizer("<answer>", add_special_tokens=False)["input_ids"][-1]
        labels = input_ids.clone()
        # only keep the loss for eos and the answer between <answer> and <endofchunk>
        for i in range(labels.shape[0]):
            answer_flag=0
            for j in range(labels.shape[1]):
                if not answer_flag:
                    if labels[i, j] == answer_token_id:
                        answer_flag=1
                    labels[i, j] = -100
                else:
                    if labels[i, j] == endofchunk_token_id:
                        answer_flag=0
                        labels[i, j] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[:, 0] = -100
        labels[labels == answer_token_id] = -100

        example = {
            "net_input":{
                "input_ids": src_item,
                "attention_masks": src_item_mask,
                "patch_images": patch_image,
                "input_len": input_len,
                "input_seq": input_seq,
                
                
            },
            "net_output":{
                "output_ids": semantic_id,
                "labels":labels,

            }
        }
        

        return example

    def process_eval_meta2id_pair(self, index):
        #def process_eval_meta2id_pair(self, index): 替代
        full_seq = self.seqs[index]
        # print("开头seqs源数据",self.seqs)
        # print("开头full_seq完整内容",full_seq)
        #print("full_seq(1)",full_seq[1])
        seq = [item[0] for item in full_seq]
        img_seq = []
        input_seq = ""
        if self.subset=="hm":
            test_len = 20
        else:
            test_len = 5
        for item in seq[-test_len:-1]:
            # p = np.random.random()
            # if p<1.1:
            #     item_seq = f"{item} <|endofchunk|> "
            # else:
            # p = np.random.random()
            # if p<0.3:
            #     image_item = Image.open(os.path.join(self.img_folder, f"0.jpg")).convert("RGB")
            # else:
            image_item = Image.open(os.path.join(self.img_folder, f"{item}.jpg")).convert("RGB")
            
            # meta_item = self.extract_meta(item)
            if self.subset=="all":
                meta_item = self.extract_meta(item)
                meta_end = self.extract_meta(seq[-1])
            elif self.subset=="netflix":
                meta_item = self.extract_meta_netflix(item)
            elif self.subset=="hm":
                meta_item = self.extract_meta_hm(item)
            img_seq.append(self.patch_resize_transform(image_item))
            # semantic_id = f"item_{item}"
            # item_seq = f"<image> {meta_item} <|endofchunk|> "
            if not self.use_semantic:
                item_seq = f"An item is described as {meta_item} <image> , can you tell me which item it is? <answer> <|endofchunk|> "
                # item_seq = f"<image> {meta_item} item_domain_{item} <|endofchunk|> "
            else:
                semantic_id = self.id2semantic[str(item)].split(",")
                semantic_id = [f"item_{id}" if i<self.len_semanticid else f"item_last_{id}" for i, id in enumerate(semantic_id)]
                semantic_id = "".join(semantic_id)
                item_seq = f"<image> {meta_item} {semantic_id} <|endofchunk|> "
            input_seq = input_seq+item_seq
        #input_seq = input_seq+f"<answer>"
        input_seq = input_seq
        #input_seq = input_seq+f"Based on the history that I provided , what is the next item recommended to the user? <answer>"
        
        # naive id
        if not self.use_semantic:
            #print("seq完整内容",seq)
            #print("full_seq完整内容",full_seq)
            semantic_id = f"item_{seq[-1]}"
            # semantic_id = f"item_domain_{seq[-1]}"
        else:
            # semantic id
            item = seq[-1]
            semantic_id = self.id2semantic[str(item)].split(",")
            # semantic_id = [f"item_{i}_{id}" for i, id in enumerate(semantic_id)]
            semantic_id = [f"item_{id}" if i<self.len_semanticid else f"item_last_{id}" for i, id in enumerate(semantic_id)]
            semantic_id = "".join(semantic_id)
        
        input_len = len(input_seq.split(" "))
        
        patch_image = torch.stack(img_seq,dim=0)

        src_text = self.tokenizer(
                input_seq,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True
            )
        
        #print("input_ids",input_ids)
        src_item = src_text["input_ids"].squeeze(0)
        src_item_mask = src_text["attention_mask"].squeeze(0)
        # print("input_ids",src_item)
        if src_item[-1] == 50280:
            src_item = torch.cat((src_item, torch.tensor([209])))
            src_item_mask = torch.cat((src_item_mask, torch.tensor([1])))
        elif src_item[-1] != 209:
            src_item = torch.cat((src_item, torch.tensor([209])))
            src_item_mask = torch.cat((src_item_mask, torch.tensor([1])))


        input_ids = src_text["input_ids"]
        #print("input_ids",input_ids)
        endofchunk_token_id = self.tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"][-1]
        answer_token_id = self.tokenizer("<answer>", add_special_tokens=False)["input_ids"][-1]
        labels = input_ids.clone()
        # only keep the loss for eos and the answer between <answer> and <endofchunk>
        for i in range(labels.shape[0]):
            answer_flag=0
            for j in range(labels.shape[1]):
                if not answer_flag:
                    if labels[i, j] == answer_token_id:
                        answer_flag=1
                    labels[i, j] = -100
                else:
                    if labels[i, j] == endofchunk_token_id:
                        answer_flag=0
                        labels[i, j] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[:, 0] = -100
        labels[labels == answer_token_id] = -100


        example = {
            "net_input":{
                "input_ids": src_item,
                "attention_masks": src_item_mask,
                "patch_images": patch_image,
                "input_len": input_len,
                "input_seq": input_seq,
                
                
            },
            "net_output":{
                "output_ids": semantic_id,
                "labels":labels,

            }
        }
        

        return example

    # def process_eval_rec_pair(self, index):
    #     full_seq = self.seqs[index]
    #     # print("开头seqs源数据",self.seqs)
    #     # print("开头full_seq完整内容",full_seq)
    #     #print("full_seq(1)",full_seq[1])
    #     seq = [item[0] for item in full_seq]
    #     img_seq = []
    #     input_seq = ""
    #     if self.subset=="hm":
    #         test_len = 20
    #     else:
    #         test_len = 5
    #     for item in seq[-test_len:-1]:
    #         # p = np.random.random()
    #         # if p<1.1:
    #         #     item_seq = f"{item} <|endofchunk|> "
    #         # else:
    #         # p = np.random.random()
    #         # if p<0.3:
    #         #     image_item = Image.open(os.path.join(self.img_folder, f"0.jpg")).convert("RGB")
    #         # else:
    #         image_item = Image.open(os.path.join(self.img_folder, f"{item}.jpg")).convert("RGB")
            
    #         # meta_item = self.extract_meta(item)
    #         if self.subset=="all":
    #             meta_item = self.extract_meta(item)
    #             meta_end = self.extract_meta(seq[-1])
    #         elif self.subset=="netflix":
    #             meta_item = self.extract_meta_netflix(item)
    #         elif self.subset=="hm":
    #             meta_item = self.extract_meta_hm(item)
    #         img_seq.append(self.patch_resize_transform(image_item))
    #         # semantic_id = f"item_{item}"
    #         # item_seq = f"<image> {meta_item} <|endofchunk|> "
    #         if not self.use_semantic:
    #             item_seq = f"<image>  {meta_item} <answer> item_{item} <|endofchunk|> "
    #             # item_seq = f"<image> {meta_item} item_domain_{item} <|endofchunk|> "
    #         else:
    #             semantic_id = self.id2semantic[str(item)].split(",")
    #             semantic_id = [f"item_{id}" if i<self.len_semanticid else f"item_last_{id}" for i, id in enumerate(semantic_id)]
    #             semantic_id = "".join(semantic_id)
    #             item_seq = f"<image> {meta_item} {semantic_id} <|endofchunk|> "
    #         input_seq = input_seq+item_seq
    #     #input_seq = input_seq+f"<answer>"
    #     input_seq = input_seq+f"What is the next item recommended to the user? <answer> "
    #     #input_seq = input_seq+f"Based on the history that I provided , what is the next item recommended to the user? <answer>"
        
    #     # naive id
    #     if not self.use_semantic:
    #         #print("seq完整内容",seq)
    #         #print("full_seq完整内容",full_seq)
    #         semantic_id = f"item_{seq[-1]}"
    #         # semantic_id = f"item_domain_{seq[-1]}"
    #     else:
    #         # semantic id
    #         item = seq[-1]
    #         semantic_id = self.id2semantic[str(item)].split(",")
    #         # semantic_id = [f"item_{i}_{id}" for i, id in enumerate(semantic_id)]
    #         semantic_id = [f"item_{id}" if i<self.len_semanticid else f"item_last_{id}" for i, id in enumerate(semantic_id)]
    #         semantic_id = "".join(semantic_id)
        
    #     input_len = len(input_seq.split(" "))
        
    #     patch_image = torch.stack(img_seq,dim=0)

    #     src_text = self.tokenizer(
    #             input_seq,
    #             return_tensors="pt",
    #             add_special_tokens=False,
    #             truncation=True
    #         )
        
    #     #print("input_ids",input_ids)
    #     src_item = src_text["input_ids"].squeeze(0)
    #     src_item_mask = src_text["attention_mask"].squeeze(0)
    #     # print("input_ids",src_item)
    #     if src_item[-1] == 50280:
    #         src_item = torch.cat((src_item, torch.tensor([209])))
    #         src_item_mask = torch.cat((src_item_mask, torch.tensor([1])))
    #     elif src_item[-1] != 209:
    #         src_item = torch.cat((src_item, torch.tensor([209])))
    #         src_item_mask = torch.cat((src_item_mask, torch.tensor([1])))


    #     input_ids = src_text["input_ids"]
    #     #print("input_ids",input_ids)
    #     endofchunk_token_id = self.tokenizer("<|endofchunk|>", add_special_tokens=False)[
    #     "input_ids"][-1]
    #     answer_token_id = self.tokenizer("<answer>", add_special_tokens=False)["input_ids"][-1]
    #     labels = input_ids.clone()
    #     # only keep the loss for eos and the answer between <answer> and <endofchunk>
    #     for i in range(labels.shape[0]):
    #         answer_flag=0
    #         for j in range(labels.shape[1]):
    #             if not answer_flag:
    #                 if labels[i, j] == answer_token_id:
    #                     answer_flag=1
    #                 labels[i, j] = -100
    #             else:
    #                 if labels[i, j] == endofchunk_token_id:
    #                     answer_flag=0
    #                     labels[i, j] = -100
    #     labels[labels == self.tokenizer.pad_token_id] = -100
    #     labels[:, 0] = -100
    #     labels[labels == answer_token_id] = -100
        
    #     example = {
    #         "net_input":{
    #             "input_ids": src_item,
    #             "attention_masks": src_item_mask,
    #             "patch_images": patch_image,
    #             "input_len": input_len,
    #             "input_seq": input_seq,
                
                
    #         },
    #         "net_output":{
    #             "output_ids": semantic_id,
    #             "labels":labels,

    #         }
    #     }
        

    #     return example

    def __len__(self):
        # print("Debug: len(self.seqs) called.")
        # #print("self.seqs:", self.seqs)  # 打印 self.seqs 的值
        # print("Call stack:")
        # traceback.print_stack()  # 打印调用堆栈
        # if self.split != "train":
        #     print(f"调试: Initializing RecDataset with task={self.tasks}, split={self.split}")
        #     print(f"调试: seqs length = {len(self.seqs)}")


        return len(self.seqs)

    def __getitem__(self, index):
        self.task = self.tasks[index]
        #print(f"self.tasks{self.tasks}self.task{self.task}")
        #print(f"调试: self.tasks = {set(self.tasks)} (showing first 10)")

        if self.task=="rec":
            # if self.split=="train":
            #     pair_samples = self.process_train_semantic_rec_pair(index)
            # else:
            #     pair_samples = self.process_eval_semantic_rec_pair(index)
            if self.split=="train":
                pair_samples = self.process_train_rec_pair(index)
            else:
                pair_samples = self.process_eval_rec_pair(index)
        elif self.task=="id2meta":
            if self.split=="train":
                pair_samples = self.process_train_id2meta_pair(index)
            else:
                pair_samples = self.process_eval_id2meta_pair(index)

        elif self.task=="meta2id":
            if self.split=="train":
                pair_samples = self.process_train_meta2id_pair(index)
            else:
                pair_samples = self.process_eval_meta2id_pair(index)

        else:
            raise KeyError("Not Supported Task")
        # if dataset is not supported
        if pair_samples is None:
            return self.__getitem__(index + 1)
        return pair_samples
    
    def collate(self, samples):
        """Merge samples of different tasks to form two mini-batches.
        Args:
            samples (List[Tuple]): samples to collate
        Returns:
            Tuple[dict]: two mini-batch containing the data of different tasks
        """

        samples_v1 = []  # containing image-text pairs
        for sample_tuple in samples:
            samples_v1.append(sample_tuple)

        res_v1 = collate_fn(
            samples_v1,
            pad_idx=self.tokenizer.pad_token_id,
            eos_idx=self.tokenizer.eos_token_id,
        )
        return res_v1
    

    # def collate(self, samples):
    #     """Merge samples of different tasks to form two mini-batches.
    #     Args:
    #         samples (List[Tuple]): samples to collate
    #     Returns:
    #         Tuple[dict]: two mini-batch containing the data of different tasks
    #     """

    #     samples_v1 = []  # containing image-text pairs
    #     for sample_tuple in samples:
    #         samples_v1.append(sample_tuple[0])

    #     res_v1 = collate_fn(
    #         samples_v1,
    #         pad_idx=self.tokenizer.pad_token_id,
    #         eos_idx=self.tokenizer.eos_token_id,
    #     )
    #     return res_v1

