import numpy as np


class WakeupModule:
    def __init__(self):
        self.past_frame_hist = None

    def _get_hist(self, img: np.ndarray):
        bins = 8
        hist_r, _ = np.histogram(
            img[:, :, 0].ravel(), bins=bins, range=[0, 256])
        hist_g, _ = np.histogram(
            img[:, :, 1].ravel(), bins=bins, range=[0, 256])
        hist_b, _ = np.histogram(
            img[:, :, 2].ravel(), bins=bins, range=[0, 256])

        hist = np.concat([hist_r, hist_g, hist_b], axis=0)
        hist = hist / (img.shape[0]*img.shape[1])

        return hist

    # img: np.ndarray, is a warped img (256,256,3)
    def is_wakeup(self, img: np.ndarray, past_img=None) -> bool:
        # compute img_hist
        hist = self._get_hist(img)

        # mse to past hist
        if past_img is None:
            past_hist = self.past_frame_hist
        else:
            past_hist = self._get_hist(past_img)

        if past_hist is None:
            ret = True
        else:
            err = -np.log(((hist - past_hist)**2).mean().item() + 1e-7)
            ret = (err < 10).item()

        # update past_hist
        if ret:
            self.past_frame_hist = hist

        return ret
