"""Queue a new reusable voice through the local API (server must be running)."""
import argparse
import json
import urllib.request

from tts.voices import COMPARISON_TEXT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='German narrator')
    parser.add_argument('--description', default='Ein professioneller, sachlicher deutscher Sprecher mit einer warmen Stimme.')
    parser.add_argument('--text', default=COMPARISON_TEXT)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    args = parser.parse_args()
    request = urllib.request.Request(args.url.rstrip('/') + '/voices',
        data=json.dumps({'name': args.name, 'description': args.description,
                         'ref_text': args.text, 'language': 'German'}).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=30) as response:
        job = json.load(response)
    print(f'Voice creation queued: {args.url}/jobs/{job["id"]}')


if __name__ == '__main__':
    main()
