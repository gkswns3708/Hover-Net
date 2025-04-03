import tqdm
from enum import Enum

class Events(Enum):
    EPOCH_STARTED = "epoch_started"
    EPOCH_COMPLETED = "epoch_completed"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STARTED = "started"
    COMPLETED = "completed"
    EXCEPTION_RAISED = "exception_raised"
    
class State(object):
    
    def __init__(self):
        # setting propagated from config
        self.logging = None
        self.log_dir = None
        self.log_info = None
        
        self.curr_epoch_step = 0 # current step in epoch
        self.curr_global_step = 0 # current global step
        self.curr_epoch = 0 # current global epoch
        
        
        self.tracked_step_output
        