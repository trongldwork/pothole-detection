import cv2
import os
import numpy as np


class Visualizer:
    def __init__(self, cfg):
        self.cfg = cfg
        np.random.seed(0)
        self.colours = np.random.randint(0, 256, size=(3, 16))
        self.writer = None

    def plot(self, frame, boxes, scores, names=None):
        if len(boxes) == 0:
            return frame
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            # print(f" x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
            # raise Exception("Stop")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{scores[i]:.2f} {names[0] if names != None else 0}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (152, 13, 56), 2)
        return frame

    def set_writer(self, output_name, fps=20, size=(640, 640), encoder='XVID'):
        if self.cfg['render_size'] is not None:
            self.render_size = self.cfg['render_size']
        else:
            self.render_size = size
        if (self.cfg['render']) and (output_name is not None):
            self.writer = cv2.VideoWriter(os.path.join(self.cfg['output_dir'], f'{output_name}.mp4'),
                                          cv2.VideoWriter_fourcc(*encoder), fps, self.render_size)

    def write_frame(self, frame):
        frame = cv2.resize(frame, self.render_size)
        if self.writer is not None:
            self.writer.write(frame)

    def show(self, img, title='Demo'):
        cv2.imshow(title, img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            quit()
        if key & 0xFF == ord('s'):  # stop
            cv2.waitKey(0)
