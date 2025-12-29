# import sys
import numpy as np
from rknnlite.api import RKNNLite

MODEL_PATH = "models/zipformer/encoder-epoch-99-avg-1.rknn"

def debug():
    print(f"📂 加载模型: {MODEL_PATH}")
    rknn = RKNNLite(verbose=False)
    rknn.load_rknn(MODEL_PATH)
    rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

    # 1. 暴力内省：看看 runtime 到底有哪些属性
    print("\n" + "="*40)
    print("🔍 Runtime 对象属性扫描")
    print("="*40)
    # 打印所有非私有属性，看看有没有 GetInputInfo 之类的
    attrs = [x for x in dir(rknn.rknn_runtime) if not x.startswith('__')]
    print(f"可用属性/方法: {attrs}")

    # 2. 投石问路：通过报错反推输入数量
    print("\n" + "="*40)
    print("🧪 输入数量压力测试")
    print("="*40)
    
    # 构造一些虚拟输入
    # 假设 Input 0 是特征 [1, 50, 80]
    dummy_feat = np.zeros((1, 50, 80), dtype=np.float32)
    # 假设 Input 1 是长度 [1, 1]
    dummy_len = np.zeros((1, 1), dtype=np.int32)
    # 假设 Input 2 是未知 [1, 1] (随便猜)
    # dummy_extra = np.zeros((1, 1), dtype=np.float32)
    
    inputs_list = []
    for i in range(1, 6): # 尝试 1 到 5 个输入
        inputs_list.append(dummy_feat if i==1 else dummy_len) # 只是为了凑数
        print(f"👉 尝试传入 {i} 个输入...", end="")
        try:
            rknn.inference(inputs=inputs_list)
            print(" ✅ 成功！(这就是正确的数量)")
            break
        except Exception as e:
            err_msg = str(e)
            if "list index out of range" in err_msg:
                print(f" ❌ 失败: 数量不够 (模型需要 > {i})")
            elif "need 2dims" in err_msg:
                 print(f" ⚠️ 数量够了，但形状/维度不对 (说明模型需要 {i} 个输入)")
                 # 这里可以尝试提取更多信息
                 break
            elif "Mismatch" in err_msg or "buffer" in err_msg:
                 print(f" ⚠️ 数量够了，但形状不匹配 (说明模型需要 {i} 个输入)")
                 break
            else:
                print(f" ❌ 其他错误: {err_msg}")

    print("\n调试结束。请把上面的【可用属性】和【尝试结果】发给我。")
    rknn.release()

if __name__ == "__main__":
    debug()