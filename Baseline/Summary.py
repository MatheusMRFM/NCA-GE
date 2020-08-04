import numpy as np
import tensorflow as tf
import Constants

class Summary():
    def __init__(self, writer, mode):
        self.mean_loss = []
        self.reg_loss = []
        self.last_write = 0
        self.mode = mode
        self.writer = writer

    def add_info(self, mean_loss, reg_loss):
        self.mean_loss.append(mean_loss)
        self.reg_loss.append(reg_loss)
        #self,write(count, mode)

    def reset(self):
        self.mean_loss = []
        self.reg_loss = []

    def write(self, count, mode):
        if count - self.last_write >= Constants.SUMMARY_INTERVAL or self.mode != mode:
            self.last_write = count
            mean_loss = np.mean(self.mean_loss)
            reg_loss = np.mean(self.reg_loss)
            #print("Main Loss = ",mean_loss, " --- Reg Loss = ", reg_loss)
            summary = tf.Summary()
            if self.mode == Constants.TRAIN:
                name = Constants.SUMMARY_NAME + '/Train_'
            else:
                name = Constants.SUMMARY_NAME + '/Test_'
            summary.value.add(tag=name+'Mean_Loss', simple_value=float(mean_loss))
            summary.value.add(tag=name+'Reg_Loss', simple_value=float(reg_loss))
            #count = float(frame_count) / float(FRAMES_IN_EPOCH)
            self.writer.add_summary(summary, count)
            self.writer.flush()
            self.reset()
            self.mode = mode
