import numpy as np
import pandas as pd
import os
import shutil

# Meta data 
tracks = pd.read_csv(r"C:\Users\lapto\Documents\Voor mij\School\Tilburg University\Master\Block 3\Thesis\Data\fma_metadata\tracks.csv", index_col=0, header=[0,1])

# Renaming the genre 'Old-Time / Historic' to 'Old-Time'
filter = tracks['track','genre_top'] == 'Old-Time / Historic'
tracks['track','genre_top'][filter] = 'Old-Time'

# Medium dataset subsetting (combination of small and medium set to get the complete set)
small_df = tracks[(tracks['set']['subset']=='small')]

# Path to songs
songs = r"C:\Users\lapto\Documents\Voor mij\School\Tilburg University\Master\Block 3\Thesis\Data\fma_small"

# List of corrupted files
corupted = ["011298", "021657", "029245", "054568", "054576", "001486", "005574", "065753", "080391", "098558", "098559", "098560", "098565", "098566", "098567", "098568", "098569", "098571", "099134", "105247", "108924", "108925", "126981", "127336", "133297", "143992"]

# Removing song file from directory
for track_id in corupted:
    # Getting the folder from the track id
    folder = track_id[:3]

    # Path to the specific song
    song_path = os.path.join(songs, "Original", folder, f"{track_id}.mp3")

    # Removing song file from directory if exist
    try:
        os.remove(song_path)
    except:
        print(f"The file {song_path} does not exist")

# Removing song file from metadata directory
for track_id in corupted:
    try:
         small_df = small_df.drop(index=int(track_id))
    except:
        print(f"The index {track_id} does not exist")


# Creating different subsets
training_set = small_df[small_df["set"]["split"]=="training"]
validation_set = small_df[small_df["set"]["split"]=="validation"]
testing_set = small_df[small_df["set"]["split"]=="test"]


# Proper formatting of track_id
def adding_zeros(track_id):
    # Padding the track id with zeros to match the MP3 file naming convention (6 digits)
    formatted_track_id = f"{track_id:06d}"
    return formatted_track_id


# Dictionary of sets
datasets = {
    'training_set': training_set,
    'validation_set': validation_set,
    'testing_set': testing_set
}

ls = []

# Moving orignal track data into subsets
for setname, dataset in datasets.items():
    set_folder = os.path.join(songs, "Ordered", setname)
    os.makedirs(set_folder, exist_ok=True)
    for row in range(len(dataset)):
        # Extracting genre name from the track
        genre_name = dataset['track','genre_top'].iloc[row]

        # Taking track_id from the index
        track_id = dataset.index[row]

        # Formatting the track id to match the naming of the MP3 files
        track_id = adding_zeros(track_id)
        
        # Getting the folder from the track id
        folder = track_id[:3]

        # Input path for the song
        input_path = os.path.join(songs, "Original", folder, f"{track_id}.mp3")
        
        # Output path for the song in the new genre-specific folder
        output_path = os.path.join(songs, setname,  f"{track_id}.mp3")

        # Check if the song exists at the source path
        if os.path.exists(input_path):
            # Copy the song to the new location
            shutil.move(input_path, output_path)
            print(f"Copied {track_id} to mp3 folder")
        else:
            ls.append(input_path)        