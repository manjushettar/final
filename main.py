from src import client

def main():
    c = client.Client()
    print(c.get_track_features('4ZtFanR9U6ndgddUvNcjcG?si=60d9868f035e426b'))
    print(c.get_user_top_tracks())
    return 0

if __name__ == '__main__':
    main()
