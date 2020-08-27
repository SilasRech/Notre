import mido
import matplotlib.pyplot as plt
import numpy as np


mid = mido.MidiFile('audio/Chords.mid')
mid_play = mid.play()

print(mid)
one_track = mid.tracks[0][677]
song = []
song_attributes = []

for i in range(10, 677):
    print(i)
    song.append(mid.tracks[0][i])
    note = getattr(mid.tracks[0][i], 'note')
    time = getattr(mid.tracks[0][i], 'time')

    song_attributes.append([note, time])


plt.plot(np.asarray(song_attributes))
plt.ylabel('some numbers')
plt.show()
check = 1