from model import *


def main():
    net = BaseCNN()
    a = torch.randn(1, 1, 16000)

    print(net.foward(a))

if __name__ == "__main__":

    main()