# -*- encoding: utf8 -*-

import json

def clean_data(filename, output_file, string, string2):
    with open(filename) as f, open(output_file, "w") as f2:
        data = json.load(f)
        a = data[string]
        b = len(a)
        for i in xrange(b):
            s = ""
            j = a[i]
            c = j['data']
            for k in xrange(len(c)):
                d = c[k]['text']
                if d!="\n" and d!="\n " and d!=" \n":
                    s = s + d
            s = s.encode('utf-8')
            f2.write(string2 + s+"\n")


if __name__ == '__main__':
    clean_data('data/train_AddToPlaylist.json', 'clean_data/AddToPlaylist.txt', 'AddToPlaylist', 'ATP : ')
    clean_data('data/train_AddToPlaylist_full.json', 'clean_data/AddToPlaylist_full.txt', 'AddToPlaylist', 'ATP : ')

    clean_data('data/train_BookRestaurant.json', 'clean_data/BookRestaurant.txt', 'BookRestaurant', 'BR : ')
    clean_data('data/train_BookRestaurant_full.json', 'clean_data/BookRestaurant_full.txt', 'BookRestaurant', 'BR : ')

    clean_data('data/train_GetWeather.json', 'clean_data/GetWeather.txt', 'GetWeather', 'WEATHER : ')
    clean_data('data/train_GetWeather_full.json', 'clean_data/GetWeather_full.txt', 'GetWeather', 'WEATHER : ')

    clean_data('data/train_PlayMusic.json', 'clean_data/PlayMusic.txt', 'PlayMusic', 'PM : ')
    clean_data('data/train_PlayMusic_full.json', 'clean_data/PlayMusic_full.txt', 'PlayMusic', 'PM : ')

    clean_data('data/train_RateBook.json', 'clean_data/RateBook.txt', 'RateBook', 'RB : ')
    clean_data('data/train_RateBook_full.json', 'clean_data/RateBook_full.txt', 'RateBook', 'RB : ')

    clean_data('data/train_SearchCreativeWork.json', 'clean_data/SearchCreativeWork.txt', 'SearchCreativeWork', 'SCW : ')
    clean_data('data/train_SearchCreativeWork_full.json', 'clean_data/SearchCreativeWork_full.txt', 'SearchCreativeWork', 'SCW : ')

    clean_data('data/train_SearchScreeningEvent.json', 'clean_data/SearchScreeningEvent.txt', 'SearchScreeningEvent', 'SSE : ')
    clean_data('data/train_SearchScreeningEvent_full.json', 'clean_data/SearchScreeningEvent_full.txt', 'SearchScreeningEvent', 'SSE : ')