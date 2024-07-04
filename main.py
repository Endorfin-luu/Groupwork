from data_preprocessing import preprocess_data
from training import train_model
from evaluation import evaluate_model
from deployment import save_model

def main():
    data = preprocess_data('data.csv')
    model, X_test, y_test = train_model(data, 'target')
    accuracy = evaluate_model(model, X_test, y_test)
    save_model(model, 'model.pkl')
    print(f'Accuracy: {accuracy}')

if __name__ == '__main__':
    main()
