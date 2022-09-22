// Making bigrams from the data and investigating them

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>


std::unordered_map<std::string, int> bigrams;

std::vector<std::string> read_csv_col(std::string filename, int col) {
    std::ifstream ifs(filename);
    std::string line;
    std::vector<std::string> colvec;
    while (std::getline(ifs, line)) {
        std::string colstr;
        int c = 0;
        for (auto ch : line) {
        if (ch == ',') {
            ++c;
            continue;
        }
        if (c == col)
            colstr += ch;
        }
        colvec.push_back(colstr);
    }
    return colvec;
}

void build_bigrams(std::vector<std::string> words) {
    for (auto word : words) {
        // add start and end tokens
        word = "*" + word + ".";
        for (int i = 0; i < word.size() - 1; ++i) {
            std::string bigram = word.substr(i, 2);
            if (bigrams.find(bigram) == bigrams.end()) 
                bigrams[bigram] = 0;
            ++bigrams[bigram];
        }
    }
}

int main() {
    std::vector<std::string> words = read_csv_col("makemore/names.csv", 0);
    build_bigrams(words);
    // Write bigrams to file
    std::ofstream ofs("makemore/bigrams.txt");
    for (auto bigram : bigrams) {
        ofs << bigram.first << " " << bigram.second << std::endl;
    }
}