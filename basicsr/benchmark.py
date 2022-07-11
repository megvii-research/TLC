
import torch
import time

def measure_inference_speed(model, input_res, max_iter=50, log_interval=10,
                            cudnn_benchmark=False):
    # set cudnn_benchmark
    if cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    try:
        batch = torch.ones(()).new_empty((1, *input_res),
            dtype=next(model.parameters()).dtype,
            device=next(model.parameters()).device)
    except StopIteration:
        batch = torch.ones(()).new_empty((1, *input_res))

    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0
    fps = 0

    # benchmark with 2000 image and take the average
    for i in range(max_iter+num_warmup):

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(batch)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(
                    f'Done image [{i + 1:<3}/ {max_iter}], '
                    f'fps: {fps:.1f} img / s, '
                    f'times per image: {1000 / fps:.1f} ms / img',
                    flush=True)

        if (i + 1) == max_iter:
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(
                f'Overall fps: {fps:.1f} img / s, '
                f'times per image: {1000 / fps:.1f} ms / img',
                flush=True)
            break
    return fps

if __name__ == '__main__':
    # h, w = 720, 1280
    # tile = 256
    # tile_overlap = 64

    # stride = tile - tile_overlap
    # h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
    # w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
    # print(len(h_idx_list) * len(w_idx_list))

    from basicsr.models.archs.mprnet_arch import MPRNet
    from basicsr.models.archs.local_arch import MPRNetLocal
    input_res = (3, 720, 1280)
    model = MPRNet().cuda()
    measure_inference_speed(model, input_res=(3, 256, 256))
    measure_inference_speed(model, input_res=input_res)
    model = MPRNetLocal(base_size=384).cuda()
    measure_inference_speed(model, input_res=input_res)
    model = MPRNetLocal(base_size=384, fast_imp=True).cuda()
    measure_inference_speed(model, input_res=input_res)