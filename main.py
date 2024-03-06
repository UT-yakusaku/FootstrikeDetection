import logging
import traceback

import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import numpy as np

import utils


def main():
    # log file
    logging.basicConfig(level=logging.INFO, filename='log.txt', filemode='w')

    # load config
    cfg = OmegaConf.load('config.yaml')
    try:
        # Load LFP data 3000 Hz 32 channels
        lfp = np.load(cfg.paths.lfp)
        strobe_TTL_idx = np.load(cfg.paths.trigger)

        # Detect onsets
        onset_frame_idx = utils.get_valid_step_idx(
            cfg, graph_preview=False
        )
        onset_frame_idx -= 2 # fix 2 frame delay

        # Align LFP data
        onsets = strobe_TTL_idx[onset_frame_idx]
        aligned_lfp = np.array([lfp[x-600:x+600] for x in onsets])

        # Plot
        fig = plt.figure(figsize=(10, 5), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(aligned_lfp[:, :, 3].T, 'darkgray', alpha=0.1, linewidth=1)
        ax.plot(np.mean(aligned_lfp[:, :, 3], axis=0), 'k', linewidth=1)
        ax.set_xticks([0, 300,  600, 900, 1200])
        ax.set_xticklabels([-200, -100, 0, 100, 200])
        ax.set_xlim(0, 1200)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Voltage (uV)')
        ax.spines[['top', 'right']].set_visible(False)
        plt.savefig('aligned_lfp.png')

    except Exception as e:
        print(e)
        print(traceback.format_exc())
        logging.error(f"Error while detecting onsets")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()