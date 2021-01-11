import os
import subprocess
import sys
import time

import torch
from torch.utils.tensorboard import SummaryWriter

USE_TENSORBOARD = True


class Logger(object):
    def __init__(self, opt):
        """Create a summary writer logging to log_dir."""
        if not os.path.exists(opt.save_dir):
            os.makedirs(opt.save_dir)
        if not os.path.exists(opt.debug_dir):
            os.makedirs(opt.debug_dir)

        time_str = time.strftime('%Y-%m-%d-%H-%M')

        args = dict((name, getattr(opt, name)) for name in dir(opt)
                    if not name.startswith('_'))
        file_name = os.path.join(opt.save_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('==> commit hash: {}\n'.format(
                # subprocess.check_output(["git", "describe"])))
                subprocess.check_output(["git", "describe", "--always"])))
            opt_file.write(f'==> torch version: {torch.__version__}\n')
            opt_file.write(f'==> cudnn version: {torch.backends.cudnn.version()}\n')
            opt_file.write('==> Cmd:\n')
            opt_file.write(str(sys.argv))
            opt_file.write('\n==> Opt:\n')
            for k, v in sorted(args.items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))

        log_dir = opt.save_dir + '/logs_{}'.format(time_str)
        if USE_TENSORBOARD:
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            if not os.path.exists(os.path.dirname(log_dir)):
                os.mkdir(os.path.dirname(log_dir))
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
        self.log = open(log_dir + '/log.txt', 'w')
        try:
            os.system('cp {}/opt.txt {}/'.format(opt.save_dir, log_dir))
        except:
            pass
        self.start_line = True

    def write(self, txt):
        if self.start_line:
            time_str = time.strftime('%Y-%m-%d-%H-%M')
            self.log.write('{}: {}'.format(time_str, txt))
        else:
            self.log.write(txt)
        self.start_line = False
        if '\n' in txt:
            self.start_line = True
            self.log.flush()

    def close(self):
        self.log.close()

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        if USE_TENSORBOARD:
            self.writer.add_scalar(tag, value, step)
