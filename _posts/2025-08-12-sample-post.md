---
layout: post
title: "Sample Post: Styling Code with Jekyll"
date: 2025-08-12
---

This post demonstrates the custom styling for code blocks. The goal is to emulate the look of a macOS window, complete with "traffic light" buttons.

Here is an example of a Python code block:

```python
import os

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))
```

And here is a shell script example:

```bash
#!/bin/bash
# A simple script to greet the user
echo "Hello, $(whoami)!"
```

The styling is handled entirely by CSS, making it easy to customize. You can find the relevant styles in `assets/style.css`.
