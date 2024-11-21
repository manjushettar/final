import os
import requests
from requests.exceptions import (
    RequestException,
    Timeout,
    ConnectionError,
    HTTPError,
    TooManyRedirects
)
from dotenv import load_dotenv
from fastapi import FastAPI
from urllib.parse import urlencode
from api_handler import SpotifyAPI


class Auth:
    def __init__(self,scope=None, show_dialog=None):
        self.response_type = 'code' 
        self.redirect_uri = 'https://manjushettar.github.io/app/index.html'
        self.client_id = 'b4c994c701684092b02511dde29b91aa'
        self.state = '31'
        self.scope = scope
        self.show_dialog = show_dialog
    

    def to_dict(self):
        return {
            'response_type': self.response_type,
            'redirect_uri': self.redirect_uri,
            'client_id': self.client_id,
            'state': self.state,
        }

class Client:
    def __init__(self):
        load_dotenv()
        self.auth = False
        
        self._auth = Auth()
        self.base_url = "https://api.spotify.com/v1"

        self.auth_flow()

    def auth_flow(self):
        if not self.authenticate():
            return

        ok, res = self.access_token()
        if not ok:
            return 
        self._token = res['access_token']
        self._token_type = res['token_type']
        self._expires = res['expires_in']
        self.header = {
            "Authorization": f"Bearer {self._token}"
        }

    def authenticate(self):
        api_url = "https://accounts.spotify.com/authorize?" + urlencode(self._auth.to_dict())  
        
        authorized = requests.get(api_url)
        if not authorized.ok:
            print("Could not auth!")
            return False
        
        self.auth = True
        return True

    def access_token(self):
        '''Get access token from Spotify API - valid for 1 hr'''
        req = {
                'grant_type': 'client_credentials',
                'client_id': os.getenv('client-id'), 
                'client_secret': os.getenv('client-secret')
            }

        header = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
       
        url = 'https://accounts.spotify.com/api/token'
        
        res = requests.post(url, data=req, headers=header)
        
        if not res.ok:
            print("Could not get access token.")
            return False, None
        
        print(res.json())
        return True, res.json()
    
    def get_related_artists(self, artist_id):
        url = f"{self.base_url}/{artist_id}/related-artists"
        
        res = requests.get(url, headers = self.header)
        if not res.ok:
            print(f"Could not get related artists: {res.status_code}")
            return False
        print(res.text)
        return True
    
    def get_track_features(self, track_id):
        url = f"{self.base_url}/audio-features/{track_id}"

        try:
            res = requests.get(url, headers = self.header)
            res.raise_for_status()
            return res.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting track features: {e}")
            return None

    def get_user_top_tracks(self, time_range='medium_term', limit=20):
        url = f"{self.base_url}/me/top/tracks"

        params = {
            'time_range':time_range,
            'limit': limit
        }

        try:
            res = requests.get(url, headers = self.header, params = params)
            res.raise_for_status()
            return res.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting track features: {e}")
            return None
