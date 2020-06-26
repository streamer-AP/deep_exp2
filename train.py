import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import json
import os
from DataSet import DataSet
from termcolor import cprint
from tqdm import tqdm
from torchvision.models import resnet50

def train(cfg,net,train_dl,eval_dl):
    Loss=torch.nn.CrossEntropyLoss()
    optim=torch.optim.Adam(net.parameters(),lr=cfg["lr"])


    best_acc=0
    writer=SummaryWriter(log_dir=cfg["log_dir"])
    if not os.path.isdir(cfg["checkpoints"]):
        os.makedirs(cfg["checkpoints"])
    for epoch in range(cfg["epochs"]):
        acc_cnt=0.
        train_sample_cnt=0
        eval_sample_cnt=0
        for batch_img,batch_target in tqdm(train_dl):
            batch_img=batch_img.cuda()
            batch_target=batch_target.cuda()
            optim.zero_grad()
            batch_predict=net(batch_img)
            loss=Loss(batch_predict,batch_target)
            loss.backward()
            optim.step()
            acc_cnt+=torch.sum(torch.argmax(batch_predict,dim=1)==batch_target)
            train_sample_cnt+=len(batch_target)
        train_acc=acc_cnt/float(train_sample_cnt)
        writer.add_scalar("train_acc",train_acc,epoch)
        cprint(f"epoch {epoch}, train_acc {train_acc}","green")
        acc_cnt=0
        for batch_img,batch_target in eval_dl:
            eval_sample_cnt+=len(batch_target)
            batch_img=batch_img.cuda()
            batch_target=batch_target.cuda()
            batch_predict=net(batch_img)
            acc_cnt+=torch.sum(torch.argmax(batch_predict,dim=1)==batch_target)
        eval_acc=acc_cnt/float(eval_sample_cnt)
        writer.add_scalar("eval_acc",eval_acc,epoch)
        cprint(f"epoch {epoch}, eval_acc {eval_acc}","green")
        if eval_acc>best_acc:
            best_acc=eval_acc
            torch.save(net.state_dict(),os.path.join(cfg["checkpoints"],f"{epoch}_{best_acc:.5f}.pt"))

if __name__ == "__main__":
    with open("config.json","r") as f:
        cfg=json.load(f)
    ds=DataSet(cfg["img_dir"])
    train_ds,eval_ds=torch.utils.data.random_split(ds,[int(0.7*len(ds)),int(0.3*len(ds))])

    train_dl=DataLoader(train_ds,batch_size=cfg["train_batch"],drop_last=True,shuffle=True,num_workers=8)
    eval_dl=DataLoader(eval_ds,batch_size=cfg["eval_batch"],drop_last=True,shuffle=False)
    net=resnet50(pretrained=False,num_classes=2,)
    net.cuda()
    net=torch.nn.DataParallel(net,device_ids=[0,1])
    train(cfg,net,train_dl,eval_dl)



