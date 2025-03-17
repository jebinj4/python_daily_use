from pytube import Playlist, YouTube
import pandas as pd

p = input("Enter URL of Playlist=")

# https://www.youtube.com/playlist?list=PLvr5U5ZSt6IzHyvSL9fo0M9NRPsTvra31

vlinks = Playlist(p)
print("Playlist Name =", vlinks.title)
print("No. of Videos =", vlinks.length)
print("Playlist ID =", vlinks.playlist_id)

# Uncomment the line below if you want to print the playlist description
# print("Playlist Description:\n", vlinks.description)

data = {'Title': [], 'Link': []}
dataframe = pd.DataFrame(data)
 
vtitles = []
for link in vlinks:
    video = YouTube(link)
    vtitles.append(video.title)

dataframe['Title'] = vtitles
dataframe['Link'] = [link for link in vlinks]

dataframe.to_excel("playlist.xlsx", index=False)
print("Playlist extracted successfully.")