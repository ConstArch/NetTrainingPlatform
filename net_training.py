from abc         import ABC, abstractmethod
from dataclasses import dataclass
from typing      import Optional

#import numpy as np
import torch


class AbstractLossApplier(ABC):
    
    @abstractmethod
    def __call__(self, net, batch):
        pass


class AbstractOptimizerFactory(ABC):
    
    @abstractmethod
    def with_parameters(self, parameters):
        pass


class IterationLogger:
    
    def __init__(self, message_sender, duration):
        self.message_sender = message_sender
        self.duration = duration
        self.count = 0
    
    def tick(self):
        self.count += 1
        if self.count % self.duration == 0:
            self.message_sender(self.count)
    
    def reset(self):
        self.count = 0


def load_all(dataset, collate_fn=torch.utils.data.default_collate):
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), collate_fn=collate_fn)
    
    return list(dataloader)[0]


@dataclass
class NetTrainer:
    
    loss_applier      : AbstractLossApplier
    optimizer_factory : AbstractOptimizerFactory
    iteration_logger  : Optional[IterationLogger] = None
    epoch_logger      : Optional[IterationLogger] = None
    
    def train(self, net, dataloader, n_epochs):
        
        net.train(True)
        
        optimizer = self.optimizer_factory.with_parameters(net.parameters())
        loss_history = []
        
        # begin outer for
        for _ in range(n_epochs):
            
            # begin inner for
            for batch in dataloader:
                
                loss = self.loss_applier(net, batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if self.iteration_logger is not None:
                    self.iteration_logger.tick()
                
            # end inner for
            
            loss_history.append(loss.item())
            
            if self.iteration_logger is not None:
                self.iteration_logger.reset()
            
            if self.epoch_logger is not None:
                self.epoch_logger.tick()
            
        # end outer for
        
        if self.epoch_logger is not None:
            self.epoch_logger.reset()
        
        net.train(False)
        
        return { 'net': net, 'loss_history': loss_history }
        
    # end NetTrainer.train
    
    def train_valid(self, net, dataloader, n_epochs, dataset_valid, metric_applier):
        
        net.train(True)
        
        optimizer = self.optimizer_factory.with_parameters(net.parameters())
        loss_history_train = []
        loss_history_valid = []
        metric_history_train = []
        metric_history_valid = []
        
        # begin outer for
        for _ in range(n_epochs):
            
            # begin inner for
            for batch in dataloader:
                
                loss = self.loss_applier(net, batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if self.iteration_logger is not None:
                    self.iteration_logger.tick()
                
            # end inner for
            
            loss_valid   = self.loss_applier(net, load_all(dataset_valid     , collate_fn=dataloader.collate_fn))
            metric_train =    metric_applier(net, load_all(dataloader.dataset, collate_fn=dataloader.collate_fn))
            metric_valid =    metric_applier(net, load_all(dataset_valid     , collate_fn=dataloader.collate_fn))
            
            loss_history_train.append(loss.item())
            loss_history_valid.append(loss_valid.item())
            metric_history_train.append(metric_train)
            metric_history_valid.append(metric_valid)
            
            if self.iteration_logger is not None:
                self.iteration_logger.reset()
            
            if self.epoch_logger is not None:
                self.epoch_logger.tick()
            
        # end outer for
        
        if self.epoch_logger is not None:
            self.epoch_logger.reset()
        
        net.train(False)
        
        return {
            'net': net,
            'loss_history_train': loss_history_train,
            'loss_history_valid': loss_history_valid,
            'metric_history_train': metric_history_train,
            'metric_history_valid': metric_history_valid
        }
        
    # end NetTrainer.train_valid
    
# end NetTrainer
