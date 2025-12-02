import joblib
import numpy as np

MIN_EXP = -32


def max_shift_within_range(arr, bits: int = 8) -> int:
    arr = np.asarray(arr)

    if np.all(arr == 0):
        return 0

    nonzero = arr[arr != 0]
    limits = []

    min_val = -(2 ** (bits - 1))
    max_val = (2 ** (bits - 1)) - 1

    for n in nonzero:
        if n > 0:
            # n * 2^i <= max => i <= log2(max / n)
            limits.append(np.log2(max_val / n))
        else:
            # n * 2^i >= min => i <= log2(|min| / |n|)
            limits.append(np.log2(abs(min_val) / abs(n)))

    i = int(np.floor(min(limits)))
    return max(-32, min(32, i))


def dump_gmms(filenames: list[str], outnames: list[str]):
    log_consts_acc = []
    means_acc = []
    inv_covs_acc = []
    K = -1
    D = -1
    for filename in filenames:
        gmm = joblib.load(filename)
        weights = gmm.weights_
        means = gmm.means_
        covariances = gmm.covariances_
        K, D = means.shape

        inv_covs = 1.0 / covariances  # 1 / sigma^2
        log_consts = np.log(weights) - 0.5 * np.sum(
            np.log(2 * np.pi * covariances), axis=1
        )

        log_consts_acc.append(log_consts)
        means_acc.append(means)
        inv_covs_acc.append(inv_covs)

    q_log_consts = max_shift_within_range(np.concatenate(log_consts_acc), bits=16)
    q_means = max_shift_within_range(np.concatenate(means_acc), bits=8)
    q_inv_covs = max_shift_within_range(np.concatenate(inv_covs_acc), bits=32)

    with open("models/gmm_params.inc", "w") as f:
        for i in range(len(filenames)):
            means = means_acc[i]
            log_consts = log_consts_acc[i]
            inv_covs = inv_covs_acc[i]
            means_quant = np.round(means * (2**q_means)).astype(np.int8)
            log_consts_quant = np.round(log_consts * (2**q_log_consts)).astype(np.int16)
            inv_covs_quant = np.round(inv_covs * (2**q_inv_covs)).astype(np.int32)

            if i == 0:
                f.write("#include <stdint.h>\n\n")
                f.write(f"#define K {K}\n")
                f.write(f"#define D {D}\n\n")
                f.write(f"#define Q_LOG_CONSTS {q_log_consts}\n")
                f.write(f"#define Q_MEANS {q_means}\n")
                f.write(f"#define Q_INV_COVS {q_inv_covs}\n\n")

            f.write(f"int16_t {outnames[i]}_log_consts[K] = {{\n")
            f.write(",\n".join(map(str, log_consts_quant)))
            f.write("\n};\n\n")

            f.write(f"int8_t {outnames[i]}_means[K][D] = {{\n")
            for k in range(K):
                f.write("{ " + ", ".join(map(str, means_quant[k])) + " }")
                if k < K - 1:
                    f.write(",\n")
            f.write("\n};\n\n")

            f.write(f"int32_t {outnames[i]}_inv_covs[K][D] = {{\n")
            for k in range(K):
                f.write("{ " + ", ".join(map(str, inv_covs_quant[k])) + " }")
                if k < K - 1:
                    f.write(",\n")
            f.write("\n};\n\n")

            f.write(f"double {outnames[i]}_log_consts_d[K] = {{\n")
            f.write(",\n".join(map(str, log_consts)))
            f.write("\n};\n\n")

            f.write(f"double {outnames[i]}_means_d[K][D] = {{\n")
            for k in range(K):
                f.write("{ " + ", ".join(map(str, means[k])) + " }")
                if k < K - 1:
                    f.write(",\n")
            f.write("\n};\n\n")

            f.write(f"double {outnames[i]}_inv_covs_d[K][D] = {{\n")
            for k in range(K):
                f.write("{ " + ", ".join(map(str, inv_covs[k])) + " }")
                if k < K - 1:
                    f.write(",\n")
            f.write("\n};\n\n")


if __name__ == "__main__":
    dump_gmms(["models/ubm_gmm.joblib", "models/target_gmm.joblib"], ["ubm", "target"])
