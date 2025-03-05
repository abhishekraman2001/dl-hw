from homework.train import train
def main():
    train(
        model_name="mlp_deep_residual",
        num_epoch=20,
        lr= 1e-3,
        hidden_dim=128,
    )

if __name__ == '__main__':
    main()