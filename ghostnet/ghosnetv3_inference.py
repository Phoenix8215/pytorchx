import torch
import torch.nn as nn
import torch.onnx
import onnxsim
import onnx
import struct
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ghostnetv3 import ghostnetv3

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def export_weight(model):
    current_path = os.path.dirname(__file__)
    f = open(current_path + "ghostnetv3.weights", 'w')
    f.write("{}\n".format(len(model.state_dict().keys())))

    for k, v in model.state_dict().items():
        print('exporting ... {}: {}'.format(k, v.shape))

        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")

    f.close()

def export_onnx(input, model):
    current_path = os.path.dirname(__file__)
    file = current_path + "ghostnetv3.onnx"
    torch.onnx.export(
        model=model, 
        args=(input,),
        f=file,
        input_names=["input0"],
        output_names=["output0"],
        opset_version=13
    )
    print("Finished ONNX export")

    model_onnx = onnx.load(file)
    onnx.checker.check_model(model_onnx)

    print(f"Simplifying with onnx-simplifier {onnxsim.__version__}...")
    model_onnx, check = onnxsim.simplify(model_onnx)
    assert check, "Simplification check failed"
    onnx.save(model_onnx, file)

def eval_model(input, model):
    output = model(input)
    print("------from inference------")
    print(input)
    print(output)

if __name__ == "__main__":
    setup_seed(1)
    
    model = ghostnetv3(width=1.0)
    model.eval()
    
    input = torch.randn(32, 3, 320, 256)

    export_weight(model)

    export_onnx(input, model)

    eval_model(input, model)