Features:
- Add moar APIs
- Multithread the server [done]
- Make port configurable [done]
- Add proper debug logging
- Add TLS for the socket

Bugs:
- Server exits when the client disconnects [done]
- Client dlsym resolution needs to switch to a hashmap or trie-based lookup table instead of linear strcmp [done]
- Enable TCP_NODELAY/prevent fragmentation across requests.
  - This one might be a little trickier than it seems as we want to ideally send one packet per request and disabling Nagle's algorithm would fragment within requests.
