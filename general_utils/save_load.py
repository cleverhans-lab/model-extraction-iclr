import pickle


def save_obj(obj, file: str):
    with open(file, 'wb') as output:
        pickle.dump(obj=obj, file=output, protocol=pickle.HIGHEST_PROTOCOL)


def load_obj(file: str):
    with open(file, 'rb') as input:
        return pickle.load(file=input)
