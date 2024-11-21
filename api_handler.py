import logging
import requests
from requests.exceptions import (
    RequestException,
    Timeout,
    ConnectionError,
    HTTPError,
    TooManyRedirects
)
from typing import Optional, Dict, Any

class SpotifyAPI:
    def __init__(self, base_url:str):
        self.base_url = base_url
        self.logger = logging.getLogger(__name__)

    def make_request(
        self, 
        endpoint: str, 
        method: str = 'GET', 
        params: Optional[Dict] = None, 
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request to API with error handling
        
        Args:
            endpoint: API endpoint
            method: HTTP method (GET, POST, etc)
            params: URL parameters
            data: Request body
            headers: Request headers
        
        Returns:
            Response data or error dict
        """
        url = f"{self.base_url}/{endpoint}"

        try:
            response = requests.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=headers,
            )
            
            response.raise_for_status()
            
            return {
                'success': True,
                'data': response.json(),
                'status_code': response.status_code
            }

        except Timeout:
            self.logger.error(f"Request timed out: {url}")
            return {
                'success': False,
                'error': 'Request timed out',
                'status_code': 408
            }

        except ConnectionError:
            self.logger.error(f"Connection failed: {url}")
            return {
                'success': False,
                'error': 'Connection failed',
                'status_code': 503
            }

        except HTTPError as e:
            self.logger.error(f"HTTP Error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'status_code': e.response.status_code
            }

        except TooManyRedirects:
            self.logger.error(f"Too many redirects: {url}")
            return {
                'success': False,
                'error': 'Too many redirects',
                'status_code': 310
            }

        except RequestException as e:
            self.logger.error(f"Request failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'status_code': 500
            }

