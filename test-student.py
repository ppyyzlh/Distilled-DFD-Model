import argparse
import yaml
from wrapper.TestModelWrapper import TestModelWrapper



def test(wrapper):
    test_loss, test_acc = wrapper.test()
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc.item() * 100:.2f}%') # 打印测试结果

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/dev/test_student.yaml', help='path to the config file')
    args = parser.parse_args()
    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    wrapper = TestModelWrapper(config)
    test(wrapper)