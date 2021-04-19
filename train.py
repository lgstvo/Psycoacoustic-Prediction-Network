import model
from argparse import ArgumentParser

def parser_arguments():
    parser = ArgumentParser()

    parser.add_argument('--solver', default='SGDM')
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--lr', default=1e-3)
    parser.add_argument('--l2reg', default=1e-4)
    parser.add_argument('--epochs', default=70)
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--validation_data', default=11830)
    parser.add_argument('--validation_frequency', default=1107)
    parser.add_argument('--gradient_threshold', default='l2')

    args = parser_args()
    return args

def train(args):


if __name__ == "__main__":


    args = parser_arguments()
    train(args)