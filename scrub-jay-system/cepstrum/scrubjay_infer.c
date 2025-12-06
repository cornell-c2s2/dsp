//  scrubjay_infer.c  –  compile with C11
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <math.h>
#include <aubio/aubio.h>
#include <onnxruntime_c_api.h>

#define N_MFCC 20
#define FEAT_DIM (N_MFCC * 2) // mean + std
#define WIN_SIZE 2048
#define HOP_SIZE (WIN_SIZE / 2)
#define N_FILTERS 40
#define AUDIO_DIR "testing"
#define MODEL_PATH "scrubjay_svm.onnx"

// ---------- MFCC helpers ----------------------------------------------------
static int mfcc_stats(const char *path, float out[FEAT_DIM])
{
    aubio_source_t *src = new_aubio_source(path, 0, HOP_SIZE);
    if (!src)
    {
        fprintf(stderr, "Cannot open %s\n", path);
        return -1;
    }

    uint_t sr = aubio_source_get_samplerate(src);
    aubio_pvoc_t *pv = new_aubio_pvoc(WIN_SIZE, HOP_SIZE);
    aubio_mfcc_t *mf = new_aubio_mfcc(WIN_SIZE, N_FILTERS, N_MFCC, sr);

    fvec_t *in = new_fvec(HOP_SIZE);
    cvec_t *spec = new_cvec(WIN_SIZE);
    fvec_t *mfcc = new_fvec(N_MFCC);

    double sum[N_MFCC] = {0.0}, sq[N_MFCC] = {0.0};
    uint64_t frames = 0, read = 0;

    do
    {
        aubio_source_do(src, in, &read);
        if (!read)
            break;
        aubio_pvoc_do(pv, in, spec);
        aubio_mfcc_do(mf, spec, mfcc);
        for (uint_t i = 0; i < N_MFCC; ++i)
        {
            double v = mfcc->data[i];
            sum[i] += v;
            sq[i] += v * v;
        }
        ++frames;
    } while (read == HOP_SIZE);

    if (frames == 0)
    {
        fprintf(stderr, "Empty file %s\n", path);
        return -1;
    }

    for (uint_t i = 0; i < N_MFCC; ++i)
    {
        double mean = sum[i] / frames;
        double var = sq[i] / frames - mean * mean;
        out[i] = (float)mean;
        out[i + N_MFCC] = (float)sqrtf(var > 0 ? var : 0);
    }

    // cleanup
    del_aubio_mfcc(mf);
    del_aubio_pvoc(pv);
    del_aubio_source(src);
    del_cvec(spec);
    del_fvec(in);
    del_fvec(mfcc);
    return 0;
}

// ---------- ONNX Runtime helpers --------------------------------------------
static OrtEnv *env;
static OrtSession *session;
static const OrtApi *api;

static void ort_check(OrtStatus *st)
{
    if (st)
    {
        fprintf(stderr, "ONNXRuntime error: %s\n", api->GetErrorMessage(st));
        exit(1);
    }
}

static void ort_init(void)
{
    api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    ort_check(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "scrubjay", &env));

    OrtSessionOptions *opt;
    ort_check(api->CreateSessionOptions(&opt));
    ort_check(api->CreateSession(env, MODEL_PATH, opt, &session));
    api->ReleaseSessionOptions(opt);
}

// returns 0/1 label and writes probability into *conf
static int ort_predict(const float feat[FEAT_DIM], float *conf)
{
    // input tensor
    OrtMemoryInfo *mem;
    ort_check(api->CreateCpuMemoryInfo(
        OrtArenaAllocator, OrtMemTypeDefault, &mem));

    int64_t shape[2] = {1, FEAT_DIM};
    OrtValue *input;
    ort_check(api->CreateTensorWithDataAsOrtValue(
        mem, (void *)feat, sizeof(float) * FEAT_DIM,
        shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input));
    api->ReleaseMemoryInfo(mem);

    const char *in_names[] = {"input"};
    const char *out_names[] = {"output_label", "output_probability"}; // names chosen by skl2onnx

    OrtValue *outputs[2] = {NULL, NULL};
    ort_check(api->Run(session, NULL,
                       in_names, &input, 1,
                       out_names, 2, outputs));

    // read outputs
    int64_t *lbl_ptr;
    ort_check(api->GetTensorMutableData(outputs[0], (void **)&lbl_ptr));
    float *probs;
    ort_check(api->GetTensorMutableData(outputs[1], (void **)&probs));

    int label = (int)lbl_ptr[0];
    *conf = probs[label];

    // tidy
    api->ReleaseValue(outputs[0]);
    api->ReleaseValue(outputs[1]);
    api->ReleaseValue(input);
    return label;
}

// ---------- main -------------------------------------------------------------
int main(void)
{
    ort_init();
    printf("Loaded model %s\n\n", MODEL_PATH);

    DIR *d = opendir(AUDIO_DIR);
    if (!d)
    {
        perror("opendir");
        return 1;
    }

    struct dirent *de;
    char path[1024];
    while ((de = readdir(d)))
    {
        if (de->d_type != DT_REG)
            continue;
        const char *ext = strrchr(de->d_name, '.');
        if (!ext || strcasecmp(ext, ".wav") * strcasecmp(ext, ".flac") * strcasecmp(ext, ".mp3") != 0)
            continue;

        snprintf(path, sizeof(path), "%s/%s", AUDIO_DIR, de->d_name);
        float feat[FEAT_DIM];
        if (mfcc_stats(path, feat) != 0)
            continue;

        float conf;
        int pred = ort_predict(feat, &conf);
        printf("%-25s → %s (confidence %.2f)\n",
               de->d_name,
               pred ? "Scrub Jay ✅" : "Not Scrub Jay ❌",
               conf);
    }
    closedir(d);
    api->ReleaseSession(session);
    api->ReleaseEnv(env);
    return 0;
}
