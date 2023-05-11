import numpy as np
from skimage import filters


def singleRetinex(img: np.ndarray, sigma: int) -> np.ndarray:
    img[img == 0] = 0.001
    return np.log10(img) - np.log10(filters.gaussian(img, sigma))


def AMSR(img: np.ndarray) -> np.ndarray:
    sigma = [15, 80, 250]
    # calculate the luminence component, Y
    Y = 0.299 * img[0:, :, :] + 0.587 * img[:, 0:, :] + 0.114 * img[:, :, 0:]

    R_SSR = [singleRetinex(img, s) for s in sigma]

    percentile_99 = [np.percentile(img, 99) for img in R_SSR]
    percentile_1 = [np.percentile(img, 1) for img in R_SSR]

    Y_SSR = [np.zeros(img.shape) for img in R_SSR]
    rounder = np.vectorize(
        lambda px, p_99, p_1: 255
        if px > p_99
        else 0
        if px < p_1
        else 255 * (px - p_1) / (p_99 - p_1)
    )

    Y_SSR = [
        rounder(R_SSR, p_99, p_1)
        for R_SSR, p_99, p_1 in zip(R_SSR, percentile_99, percentile_1)
    ]

    mu = [32, 96, 160, 224]
    sigma = 32

    p0 = np.ones(Y.shape)
    p1 = np.exp(-(Y - mu[1]) / (2 * sigma**2))
    p2 = np.exp(-(Y - mu[2]) / (2 * sigma**2))
    p3 = np.maximum(
        np.exp(-Y - mu[3]) / (2 * sigma**2), np.exp(-(Y - mu[0]) / (2 * sigma**2))
    )
    probabilites = [p0, p1, p2, p3]

    # The weight is the number of times a pixel has occured in the current distribution over the total number of the times
    # That the pixel has occured in all distributions
    sum_array = np.sum([probability for probability in probabilites])
    weights = [np.divide(probability, sum_array) for probability in probabilites]

    Y_AMSR = np.zeros(img.shape)
    Y_AMSR = weights[0] * Y + sum(
        [weight * Y_SSR for weight, Y_SSR in zip(weights[1:], Y_SSR)]
    )

    R_reconstructed = (
        1
        / 2
        * (
            np.divide(Y_AMSR, Y) * (np.add(Y[0:, :, :], Y))
            + np.subtract(Y[0:, :, :], Y)
        )
    )
    G_reconstructed = (
        1
        / 2
        * (
            np.divide(Y_AMSR, Y) * (np.add(Y[:, 0:, :], Y))
            + np.subtract(Y[:, 0:, :], Y)
        )
    )
    B_reconstructed = (
        1
        / 2
        * (
            np.divide(Y_AMSR, Y) * (np.add(Y[:, :, 0:], Y))
            + np.subtract(Y[:, :, 0:], Y)
        )
    )

    img_reconstructed = np.zeros(img.shape)
    img_reconstructed[0:, :, :] = R_reconstructed
    img_reconstructed[:, 0:, :] = G_reconstructed
    img_reconstructed[:, :, 0:] = B_reconstructed

    img_reconstructed = img_reconstructed[::-1, ::-1]  # Rotate 180 degrees
    return img_reconstructed.astype(np.uint8)
