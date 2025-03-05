from homework.train import train
def main():
    train(
        model_name="linear",
        num_epoch=50,
        lr=1e-4,
    )

if __name__ == '__main__':
    main()