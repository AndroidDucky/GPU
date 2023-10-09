#ifndef WORDCOUNT
#define WORDCOUNT

#define MAX_WORDS 30000
#define MAX_SIZE 100

typedef struct structWord
{
    char *word;
    unsigned int count;
}wordStruct;

typedef struct structWord wordStruct;

int countCmp(const wordStruct *w1, const wordStruct *w2);
int wordCmp(const wordStruct *w1, const wordStruct *w2);
void countSortPrint(const wordStruct *words);
int addUnique(const char *w, wordStruct *words);
void cleanWords(char *w);
void removeSpecialCharacter(char *w, const int pos);
void initWords(wordStruct *words);
void wordSortPrint(const wordStruct *words);
#endif