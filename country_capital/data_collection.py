import random
import pandas as pd
from transformers import AutoTokenizer
from collections import defaultdict
import json
import os


def derive_country_capital(path="country-list.csv"):
    df = pd.read_csv(path)
    countries, capitals = df['country'].tolist(), df['capital'].tolist()
    filtered_data = [(country, capital) for country, capital in zip(countries, capitals)
                     if country.isalpha() and capital.isalpha()]
    countries, capitals = zip(*filtered_data)
    return countries, capitals


def load_tokenizer(model_name="EleutherAI/pythia-1.4b-deduped"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def capital_to_country(countries, capitals, tokenizer):
    length_2_capital = defaultdict(list)
    length_2_country = defaultdict(list)
    capital_2_country = {}
    country_2_capital = {}

    for country, capital in zip(countries, capitals):
        capital_length = len(tokenizer(capital)['input_ids'])
        country_length = len(tokenizer(country)['input_ids'])

        length_2_capital[capital_length].append(capital)
        length_2_country[country_length].append(country)

        capital_2_country[capital] = country
        country_2_capital[country] = capital

    for length, values in length_2_capital.items():
        print(length, "==", len(values))
    return length_2_capital, length_2_country, capital_2_country, country_2_capital


def generate_data_wrong_logic(countries, capitals, capital_2_country, length_2_capital, tokenizer,
                              save_path="data_wrong_logic"):
    template = "If {} corresponds to {}, then {} corresponds to"
    clean, corrupt = [], []
    labels, corrupt_labels = [], []

    for country, capital in zip(countries, capitals):
        random_capital = random.choice(capitals)
        while capital == random_capital:
            random_capital = random.choice(capitals)
        true_country = capital_2_country[random_capital]

        capital_length = len(tokenizer(capital)['input_ids'])
        for corrupt_capital in length_2_capital[capital_length]:
            if corrupt_capital != capital:
                clean.append(template.format(country, capital, true_country))
                corrupt.append(template.format(country, corrupt_capital, true_country))
                labels.append(random_capital)
                corrupt_labels.append(corrupt_capital)

    dic = {'clean': clean, 'corrupt': corrupt, "predict": labels, 'corrupt_labels': corrupt_labels}
    df = pd.DataFrame(dic)
    df.to_csv(os.path.join(save_path, "country_capital_reasoning.csv"), index=False)

    data = []
    for clean_, corrupt_, label, corrupt_label in zip(clean, corrupt, labels, corrupt_labels):
        temp_data = {"clean": clean_, "corrupt": corrupt_, "label": label, 'corrupt_label': corrupt_label}
        data.append(temp_data)

    max_ = 0
    for data_ in data:
        max_ = max(max_, len(tokenizer(data_['clean'])['input_ids']))
    print(f"max length of tokens is {max_}")
    return data


def generate_data_correct_logic(countries, capitals, country_2_capital, length_2_country, tokenizer,
                                save_path="data_correct_logic"):
    template = "If {} corresponds to {}, then {} corresponds to"
    clean, corrupt = [], []
    labels, corrupt_labels = [], []

    for country, capital in zip(countries, capitals):
        random_country = random.choice(countries)
        while random_country == country:
            random_country = random.choice(countries)

        random_capital = country_2_capital[random_country]

        country_length = len(tokenizer(random_country)['input_ids'])
        for corrupt_country in length_2_country[country_length]:
            if corrupt_country != random_country and corrupt_country != country:
                clean.append(template.format(country, capital, random_country))
                corrupt.append(template.format(country, capital, corrupt_country))
                labels.append(random_capital)
                corrupt_labels.append(country_2_capital[corrupt_country])

    dic = {'clean': clean, 'corrupt': corrupt, "predict": labels, 'corrupt_labels': corrupt_labels}
    df = pd.DataFrame(dic)
    df.to_csv(os.path.join(save_path, "country_capital_reasoning.csv"), index=False)

    data = []
    for clean_, corrupt_, label, corrupt_label in zip(clean, corrupt, labels, corrupt_labels):
        temp_data = {"clean": clean_, "corrupt": corrupt_, "label": label, 'corrupt_label': corrupt_label}
        data.append(temp_data)

    max_ = 0
    for data_ in data:
        max_ = max(max_, len(tokenizer(data_['clean'])['input_ids']))
    print(f"max length of tokens is {max_}")
    return data


def train_test_split(data, save_path="data_wrong_logic"):
    random.shuffle(data)
    train_idx = int(len(data) * 0.8)
    train_data, test_data = data[:train_idx], data[train_idx:]

    with open(os.path.join(save_path, "country_capital_train.jsonl"), 'w') as f:
        for temp_data in train_data:
            f.write(json.dumps(temp_data) + "\n")

    with open(os.path.join(save_path, "country_capital_test.jsonl"), 'w') as f:
        for temp_data in test_data:
            f.write(json.dumps(temp_data) + "\n")


if __name__ == '__main__':
    countries, capitals = derive_country_capital()
    tokenizer = load_tokenizer()
    length_2_capital, length_2_country, capital_2_country, country_2_capital = capital_to_country(countries, capitals,
                                                                                                  tokenizer)

    save_path = "data_wrong_logic"
    data = generate_data_wrong_logic(countries, capitals, capital_2_country, length_2_capital, tokenizer,
                                     save_path=save_path)
    train_test_split(data, save_path)

    save_path = "data_correct_logic"
    data = generate_data_correct_logic(countries, capitals, country_2_capital, length_2_country, tokenizer, save_path)
    train_test_split(data, save_path)
