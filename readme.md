# 256 color

Generate a full 256 palette from base16 your colors. [Terminals should do this by default](https://gist.github.com/jake-stewart/0a8ea46159a7da2c808e5be2177e1783).

https://github.com/user-attachments/assets/d44f3955-e6b7-4942-9b91-3f9dc907e0dc

### Usage

```sh
# view a theme
python3 color256.py themes/century.dark.txt

# apply a theme temporarily
python3 color256.py --apply themes/century.dark.txt

# generate a theme
python3 color256.py --generate kitty themes/century.dark.txt > century.dark.conf
```

Use `python3 color256.py --help` for more options and supported terminals.

### Public Domain

All code in this repository is public domain. Use it, modify it, sell it. You do not need to credit me in any way.
