---
layout: post
title:  "🛠️ Problem: Pylint E1101 Module 'torch' has no 'from_numpy' member"
date:   2021-10-14 16:18:12 -0700
categories: Troubleshooting, Code
---

## Problem: 

See this problem [Pylint E1101 Module 'torch' has no 'from_numpy' member](https://github.com/pytorch/pytorch/issues/701)

## 1. Solution for VSCode

For those using vscode, add to user settings:


```json
"python.linting.pylintArgs": [
"--errors-only",
"--generated-members=numpy.* ,torch.* ,cv2.* , cv.*"
]
```

## 2. FYI:

Depending on your platform, the user settings file is located here:

- Windows: `%APPDATA%\Code\User\settings.json`
- macOS: `$HOME/Library/Application Support/Code/User/settings.json`
- Linux: `$HOME/.config/Code/User/settings.json`