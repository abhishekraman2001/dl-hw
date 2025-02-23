from homework.train import train
def main():
    train(
        model_name="linear",
        num_epoch=10,
        lr=1e-3,
    )

if __name__ == '__main__':
    main()