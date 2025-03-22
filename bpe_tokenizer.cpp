#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

auto start_time = 0;

// Хеш для пары<string,string> для unordered_map
struct PairHash {
    size_t operator()(const pair<string, string> &p) const {
        // комбинируем хеши для first и second с помощью побитового сдвига
        return hash<string>()(p.first) ^ (hash<string>()(p.second) << 1);
    }
};

// Разбивает строку по пробелам и возвращает вектор токенов
vector<string> split(const string &line) {
    istringstream iss(line);
    vector<string> tokens((istream_iterator<string>(iss)), istream_iterator<string>());
    return tokens;
}

// Объединяет вектор токенов в одну строку с разделением пробелом
string join(const vector<string> &tokens) {
    ostringstream oss;
    for (size_t i = 0; i < tokens.size(); i++) {
        if (i > 0) oss << " ";
        oss << tokens[i];
    }
    return oss.str();
}

// Byte Pair Encoding
vector<string> bpe_tokenize(const vector<string> &corpus, int num_merges) {
    // Представляем каждое слово в виде вектора токенов
    vector<vector<string> > words;
    words.reserve(corpus.size());
    for (const string &line : corpus) {
        words.push_back(split(line));
    }

    // Выполняем num_merges итераций слияния
    for (int merge_iter = 0; merge_iter < num_merges; merge_iter++) {
        // Подсчитываем частоты пар токенов по всему корпусу.
        unordered_map<pair<string, string>, int, PairHash> pair_freq;

        // Перебираем каждое слово (в виде вектора токенов), считаем все соседние пары токенов
        for (const auto &token_vec : words) {
            for (size_t i = 0; i + 1 < token_vec.size(); i++) {
                pair<string, string> p(token_vec[i], token_vec[i + 1]);
                pair_freq[p] += 1;
            }
        }

        if (merge_iter % 100 == 0)
            cout << "Итерация " << merge_iter + 1 << ": найдено " << pair_freq.size() << " уникальных пар. (" << double(clock() - start_time) / CLOCKS_PER_SEC
                 << " сек)" << endl;

        // Если не найдено ни одной пары, выходим из цикла
        if (pair_freq.empty()) break;

        // Находим пару с максимальной частотой
        pair<string, string> best_pair;
        int max_freq = 0;
        for (const auto &entry : pair_freq) {
            if (entry.second > max_freq) {
                // || entry.first.second.find("</w>") != string::npos
                if (entry.first.first.find("</w>") != string::npos || entry.first.first.find("</n>") != string::npos)
                    continue;  // Пропускаем если в паре в первом токене есть </w> или </n>
                max_freq = entry.second;
                best_pair = entry.first;
            }
        }
        // Если лучшая пара встречается 0 раз, прекращаем слияния
        if (max_freq == 0) break;

        // Формируем новый токен как конкатенацию двух токенов из лучшей пары
        string new_token = best_pair.first + best_pair.second;
        if (merge_iter % 100 == 0) cout << "Лучшая пара: " << best_pair.first << " + " << best_pair.second << " -> " << new_token << endl;

        // Обновляем каждое слово: заменяем вхождения лучшей пары на новый токен.
        for (auto &token_vec : words) {
            vector<string> new_tokens;
            new_tokens.reserve(token_vec.size());
            size_t i = 0;
            while (i < token_vec.size()) {
                // Если текущая пара совпадает с best_pair, объединяем их
                if (i < token_vec.size() - 1 && token_vec[i] == best_pair.first && token_vec[i + 1] == best_pair.second) {
                    new_tokens.push_back(new_token);
                    i += 2;  // пропускаем два токена, так как они объединены
                } else {
                    new_tokens.push_back(token_vec[i]);
                    i++;
                }
            }
            token_vec = std::move(new_tokens);
        }
        if (merge_iter % 100 == 0)
            cout << "Слияние " << merge_iter + 1 << " завершено." << " (" << double(clock() - start_time) / CLOCKS_PER_SEC << " сек)" << endl;
    }

    // Собираем итоговый токенизированный текст: каждое слово преобразуем обратно в строку
    vector<string> tokenized_corpus;
    tokenized_corpus.reserve(words.size());
    for (const auto &token_vec : words) {
        tokenized_corpus.push_back(join(token_vec));
    }

    return tokenized_corpus;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        cerr << "Использование: " << argv[0] << " input_file output_file num_merges" << endl;
        return 1;
    }

    string input_path = argv[1];
    string output_path = argv[2];
    int num_merges = atoi(argv[3]);

    // Считываем входной файл. Ожидается, что каждая строка уже содержит символы, разделённые пробелом, с маркером </w> в конце.
    ifstream infile(input_path);
    if (!infile) {
        cerr << "Ошибка при открытии входного файла: " << input_path << endl;
        return 1;
    }

    vector<string> corpus;
    string line;
    while (getline(infile, line)) {
        if (!line.empty()) {
            corpus.push_back(line);
        }
    }
    infile.close();

    cout << "Прочитано " << corpus.size() << " строк. (" << double(clock() - start_time) / CLOCKS_PER_SEC << " сек)" << endl;

    // Выполняем BPE-токенизацию
    vector<string> tokenized_text = bpe_tokenize(corpus, num_merges);

    cout << "BPE-токенизация завершена. (" << double(clock() - start_time) / CLOCKS_PER_SEC << " сек)" << endl;

    // Записываем результат в выходной файл
    ofstream outfile(output_path);
    if (!outfile) {
        cerr << "Ошибка при открытии выходного файла: " << output_path << endl;
        return 1;
    }

    for (const string &tokenized_line : tokenized_text) {
        outfile << tokenized_line << "\n";
    }
    outfile.close();

    cout << "BPE-токенизация завершена. Результат сохранён в " << output_path << " (" << double(clock() - start_time) / CLOCKS_PER_SEC << " сек)" << endl;
    return 0;
}
