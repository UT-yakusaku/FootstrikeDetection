import cv2
import serial
import sys

from time import sleep
from datetime import datetime
from omegaconf import OmegaConf


def main(cfg):
    # open camera
    cap = cv2.VideoCapture(cfg.camera_id)
    cap.set(cv2.CAP_PROP_FPS, cfg.fps)

    # set video codec
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    # open serial port
    with serial.Serial(cfg.serial_port, 115200) as ser:
        # wait for arduino to boot
        sleep(3)

        # start recording
        framecount = 0
        while (True):
            # Capture frame-by-frame
            flag = b's'
            ser.write(flag)
            ret, frame = cap.read()
            flag = b'q'
            ser.write(flag)

            # print frame count
            sys.stdout.write("\r{}".format(framecount))
            sys.stdout.flush()

            # write frame to video every 5 minutes
            if framecount % 288000 == 0:
                video_name = cfg.trial_name + '_' + \
                    datetime.now().strftime("%Y-%m-%d_%H%M") + '.avi'
                out = cv2.VideoWriter(video_name, fourcc,
                                      cfg.fps, cfg.resolution)
                sleep(1)

            framecount += 1

            # display video
            cv2.imshow('frame', frame)

            # stop recording with Q
            if cv2.waitKey(1) & 0xFF == ord('q') or framecount == 3000000:
                break
        ser.close()

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    camera_cfg = OmegaConf.load('camera_config.yaml')
    main(camera_cfg)
