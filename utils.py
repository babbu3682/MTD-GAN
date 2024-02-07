import torch
import time
import datetime
from collections import defaultdict, deque, OrderedDict
import pydicom
import numpy as np


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t", n=1):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.n = n

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(value=v, n=self.n)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")



def fix_optimizer(optimizer):
    # Optimizer Error fix...!
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()


def str2bool(value):
    value = value.lower()
    if value in ['true', '1', 'yes', 'y', 'on']:
        return True
    elif value in ['false', '0', 'no', 'n', 'off']:
        return False
    else:
        raise ValueError(f"Invalid boolean value: {value}")


def check_checkpoint_if_wrapper(model_state_dict):
    if list(model_state_dict.keys())[0].startswith('module'):
        return OrderedDict({k.replace('module.', ''): v for k, v in model_state_dict.items()}) # 'module.' 제거
    else:
        return model_state_dict


def dicom_denormalize(image, MIN_HU=-1024.0, MAX_HU=3072.0):
    # image = (image - 0.5) / 0.5           # Range -1.0 ~ 1.0   @ We do not use -1~1 range becuase there is no Tanh act.
    image = (MAX_HU - MIN_HU)*image + MIN_HU
    return image


def save_dicom(original_dcm_path, pred_output, save_path):
    # pydicom 으로 저장시 자동으로 -1024를 가하는 부분이 있기에 setting 해줘야 함.
    # pred_img's Range: -1024 ~ 3072
    pred_img = pred_output.copy()
    # print("before == ", pred_img.max(), pred_img.min(), pred_img.dtype) # before ==  2557.0 / -1024.0 / float32
    
    dcm = pydicom.dcmread(original_dcm_path)    

    intercept = dcm.RescaleIntercept
    slope     = dcm.RescaleSlope
    
    # pred_img -= np.int16(intercept)
    pred_img -= np.float32(intercept)
    pred_img = pred_img.astype(np.int16)

    if slope != 1:
        pred_img = pred_img.astype(np.float32) / slope
        pred_img = pred_img.astype(np.int16)

    dcm.PixelData = pred_img.squeeze().tobytes()
    # dcm.PixelData = pred_img.astype('uint16').squeeze().tobytes()
    dcm.save_as(save_path)
    
    # print("after == ", pred_img.max(), pred_img.min(), pred_img.dtype)  # after ==  3581 / 0 / int16
    # print(save_path)




def print_args(args):
    print('***********************************************')
    print('---------- DATA ---------------')
    print('Dataset Name:          ', args.dataset)
    print('Dataset [train] Type:  ', args.dataset_type_train)
    print('Dataset [valid] Type:  ', args.dataset_type_valid)
    print('---------- Model --------------')
    print('Resume From:           ', args.resume)
    print('Checkpoint To:         ', args.checkpoint_dir)
    print('Save       To:         ', args.save_dir)
    print('---------- Optimizer ----------')
    print('Learning Rate:         ', args.lr)
    print('Batchsize:             ', args.batch_size)
    

def print_args_test(args):
    print('***********************************************')
    print('---------- DATA -----------')
    print('Dataset Name:          ', args.dataset)
    print('Dataset [test] Type:  ', args.dataset_type_test)
    print('---------- Model --------------')
    print('Resume From:           ', args.resume)
    print('Checkpoint To:         ', args.checkpoint_dir)
    print('Save       To:         ', args.save_dir)
