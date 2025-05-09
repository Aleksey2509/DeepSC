import matplotlib.pyplot as plt


def plot_score(scores, subplot_tuple_arr):
    names = ["BLEU (" + str(i + 1) + "-grams" for i in range(len(scores))]
    SNR = [0, 3, 6, 9, 12, 15, 18]
    for subplot_tuple, score_arr, name in zip(subplot_tuple_arr, scores, names):
        plt.subplot(*subplot_tuple)
        plt.plot(SNR, score_arr, color="black", marker="*")
        plt.grid()

        plt.ylim((0, 1))

        plt.xticks([0, 6, 12, 18], ["0", "6", "12", "18"])

        plt.xlabel("SNR")
        plt.ylabel(name)


def main():

    AWGN_scores = [
        [
            0.68674125,
            0.88638925,
            0.9206304,
            0.92686051,
            0.92882299,
            0.92964108,
            0.93019256,
        ],
        [
            0.51100712,
            0.80100543,
            0.85651319,
            0.86662934,
            0.86986191,
            0.87134926,
            0.87204443,
        ],
        [
            0.39342203,
            0.72853797,
            0.79994711,
            0.81341939,
            0.81756349,
            0.81955549,
            0.82037874,
        ],
        [
            0.30638086,
            0.66285193,
            0.74696573,
            0.76283761,
            0.76803113,
            0.77026485,
            0.77128208,
        ],
    ]
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle('BLEU scores (SNR) for AWGN channel')
    subplot_tuples = [(1, 4, i) for i in range(1, 5)]
    plot_score(AWGN_scores, subplot_tuples)
    plt.show(block=False)

    Rayleigh_scores = [
        [
            0.49947797,
            0.66161904,
            0.78437804,
            0.84770151,
            0.88469034,
            0.90142844,
            0.92355523,
        ],
        [
            0.35877816,
            0.53809973,
            0.68813088,
            0.76776468,
            0.81281556,
            0.83562047,
            0.86319797,
        ],
        [
            0.29361245,
            0.46745909,
            0.62341399,
            0.70797369,
            0.75552491,
            0.7811357,
            0.81030207,
        ],
        [
            0.24924169,
            0.41381411,
            0.56936042,
            0.65532947,
            0.70384367,
            0.73081615,
            0.76039781,
        ],
    ]
    
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle("BLEU scores (SNR) for Rayleigh channel")
    subplot_tuples = [(1, 4, i + 1) for i in range(4)]
    plot_score(Rayleigh_scores, subplot_tuples)
    plt.show(block=False)

    Rayleigh_sim = [0.77019626, 0.8378761, 0.87810004, 0.91016114, 0.9233568, 0.92891645, 0.93895006]
    AWGN_sim = [0.83274305, 0.92246175, 0.9401382, 0.94345677, 0.9448516, 0.94530785, 0.94574463]
    
    plt.subplots(1, 2, figsize=(10, 4))
    SNR = [3 * i for i in range(7)]

    plt.subplot(1, 2, 1)
    plt.plot(SNR, AWGN_sim, marker="*")
    plt.title("Semantic similarity with AWGN channel")
    plt.xlabel('SNR')
    plt.ylabel('Semantic similarity')
    plt.ylim((0, 1))
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(SNR, Rayleigh_sim, marker="*")
    plt.ylim((0, 1))
    plt.title("Semantic similarity with Rayleigh channel")
    plt.xlabel('SNR')
    plt.ylabel('Semantic similarity')
    plt.grid()
    plt.show()



if __name__ == "__main__":
    main()
