# WSL Environment Setup for MBPP Code Translations

This guide explains how to set up all the required compilers and runtimes on WSL (Windows Subsystem for Linux) to run the MBPP code translation validation pipeline.

> **See also:** [README.md](README.md) | [DEVELOPMENT.md](DEVELOPMENT.md)

> ⚠️ **Disclaimer:** Package URLs and available versions may change over time. If a command fails, check the official installation docs for each runtime.

## Prerequisites

- WSL 2 with Ubuntu 22.04 or later
- Python 3.10+ (usually pre-installed)
- Anthropic API key

## Quick Setup (All Languages)

Run all commands in your WSL terminal:

```bash
# Update package lists
sudo apt update && sudo apt upgrade -y

# Install Node.js 20.x (JavaScript & TypeScript)
# Source: https://github.com/nodesource/distributions
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# Install TypeScript globally
sudo npm install -g typescript ts-node

# Install Go (apt version may be older, check with `go version`)
sudo apt install -y golang-go
# For specific version, see "Go" section below

# Install Java 17 (JDK)
sudo apt install -y openjdk-17-jdk

# Install Ruby
sudo apt install -y ruby ruby-dev

# Install Rust via rustup (REQUIRED for Cargo support)
# Source: https://rustup.rs/
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Install Python dependencies
pip install -r requirements.txt
```

> **Important:** Rust MUST be installed via rustup to get Cargo. The `apt` rust package does NOT include Cargo, which is required for external crate support (regex, num-bigint).

## Individual Language Setup

### JavaScript (Node.js)

```bash
# Install Node.js 20.x via NodeSource
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# Verify installation
node --version  # Should show v20.x.x
npm --version
```

### TypeScript

```bash
# Requires Node.js first
sudo npm install -g typescript ts-node

# Verify installation
tsc --version   # Should show Version 5.x.x
ts-node --version
```

### Go

```bash
# Option 1: Via apt (may be older version)
sudo apt install -y golang-go

# Option 2: Via snap (latest version)
sudo snap install go --classic

# Option 3: Manual installation (specific version)
GO_VERSION=1.22.0
wget https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go${GO_VERSION}.linux-amd64.tar.gz
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
source ~/.bashrc
rm go${GO_VERSION}.linux-amd64.tar.gz

# Verify installation
go version  # Should show go1.22.x or higher
```

### Java

```bash
# Install OpenJDK 17
sudo apt install -y openjdk-17-jdk

# Verify installation
java --version   # Should show openjdk 17.x.x
javac --version  # Should show javac 17.x.x
```

### Ruby

```bash
# Install Ruby via apt
sudo apt install -y ruby ruby-dev

# Verify installation
ruby --version  # Should show ruby 3.x.x
```

### Rust (with Cargo - REQUIRED)

⚠️ **CRITICAL:** You MUST install Rust via rustup to get Cargo. Do NOT use `apt install rust` as it doesn't include Cargo, which is needed for external crate support.

```bash
# Install via rustup (the ONLY recommended method)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# When prompted, select option 1 (default installation)
# Then reload shell environment
source "$HOME/.cargo/env"

# Verify installation - BOTH rustc AND cargo must be available
rustc --version  # Should show rustc 1.7x.x or higher
cargo --version  # Should show cargo 1.7x.x or higher
```

**Why Cargo is required:**
The Rust validator uses Cargo to compile code that depends on external crates:
- `regex` - for pattern matching (Python's `re` module equivalent)
- `num-bigint` - for arbitrary precision integers (Python handles this natively)

Without Cargo, tasks using these features will fail.

## Verification Script

Run this to verify all installations:

```bash
echo "=== Checking All Runtimes ==="
echo ""
echo "Node.js:    $(node --version 2>/dev/null || echo 'NOT INSTALLED')"
echo "npm:        $(npm --version 2>/dev/null || echo 'NOT INSTALLED')"
echo "TypeScript: $(tsc --version 2>/dev/null || echo 'NOT INSTALLED')"
echo "ts-node:    $(ts-node --version 2>/dev/null || echo 'NOT INSTALLED')"
echo "Go:         $(go version 2>/dev/null || echo 'NOT INSTALLED')"
echo "Java:       $(java --version 2>&1 | head -1 || echo 'NOT INSTALLED')"
echo "javac:      $(javac --version 2>&1 || echo 'NOT INSTALLED')"
echo "Ruby:       $(ruby --version 2>/dev/null || echo 'NOT INSTALLED')"
echo "Rust:       $(rustc --version 2>/dev/null || echo 'NOT INSTALLED')"
echo "Cargo:      $(cargo --version 2>/dev/null || echo 'NOT INSTALLED ⚠️ REQUIRED!')"
echo "Python:     $(python3 --version 2>/dev/null || echo 'NOT INSTALLED')"
echo ""
echo "=== Cargo Check ==="
if command -v cargo &> /dev/null; then
    echo "✅ Cargo is installed - Rust external crates will work"
else
    echo "❌ Cargo is NOT installed - Rust regex/bigint tasks will FAIL"
    echo "   Run: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
fi
```

## Python Environment Setup

```bash
# Navigate to the project directory (adjust path as needed)
cd /mnt/c/Users/YOUR_USERNAME/Documents/arb/sdtd-lts/MuSR/mbpp-code-translations

# Create virtual environment (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Set your Anthropic API key
export ANTHROPIC_API_KEY="your-api-key-here"
# Or add to ~/.bashrc for persistence:
# echo 'export ANTHROPIC_API_KEY="your-api-key-here"' >> ~/.bashrc
```

## Running the Pipeline

Once all runtimes are installed:

```bash
# Navigate to project directory
cd /mnt/c/Users/YOUR_USERNAME/Documents/arb/sdtd-lts/MuSR/mbpp-code-translations

# Test with a single sample
python3 translate_mbpp.py --language javascript --num-samples 1

# Run a specific language
python3 translate_mbpp.py --language typescript --num-samples 500 --all-splits --concurrency 16

# Run all languages
python3 translate_mbpp.py --language javascript,typescript,go,java,ruby,rust --num-samples 500 --all-splits --concurrency 16

# Retry failed tasks only
python3 translate_mbpp.py --language rust --retry-failed --max-attempts 10
```

## First-Run Notes

### Rust Crate Caching

The first time Rust validation runs, Cargo will:
1. Download the `regex`, `num-bigint`, and `num-traits` crates
2. Compile them into a cached project at `/tmp/mbpp_rust_cargo_cache/`

This takes **~30-60 seconds** on first run. Subsequent runs use the cache and are much faster (~2-3s per task).

If you see slow Rust validation on first run, this is normal!

## Troubleshooting

### Node.js version too old

```bash
# Remove old version
sudo apt remove nodejs
# Install fresh v20
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

### Rust/Cargo not found after installation

```bash
# Reload shell environment
source "$HOME/.cargo/env"
# Or restart your terminal completely
```

### Go not in PATH

```bash
# Add to PATH manually
export PATH=$PATH:/usr/local/go/bin
# Or for permanent fix
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
source ~/.bashrc
```

### Java JAVA_HOME not set

```bash
# Find Java installation
sudo update-alternatives --config java
# Set JAVA_HOME (adjust path if needed)
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
echo 'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64' >> ~/.bashrc
```

### Permission denied errors

```bash
# If npm global installs fail
sudo chown -R $USER:$USER ~/.npm
# Or use without sudo and configure npm prefix
```

### Timeouts during validation

Current timeout values in `validators/<language>.py`:
- Rust: 120s (Cargo compilation can be slow on first run)
- Go: 30s
- Java: 30s
- TypeScript: 30s (type checking overhead)
- JavaScript: 20s
- Ruby: 15s

If you see timeout errors, increase `TIMEOUT` in the relevant validator file.

### Rust tasks failing with "cargo not found"

This means Rust was installed via `apt` instead of rustup:

```bash
# Remove apt-installed rust (if any)
sudo apt remove rustc

# Install properly via rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Verify
cargo --version  # Must show version
```

### WSL not starting / Catastrophic failure

If WSL shows "Catastrophic failure" or won't start:

```powershell
# In PowerShell (Admin):
wsl --shutdown
net stop LxssManager
net start LxssManager
wsl
```

## Version Requirements

**Tested working configuration** (verified on WSL Ubuntu 2026-01-12):

| Runtime | Tested Version | Installation | Required |
|---------|---------------|--------------|----------|
| Python | 3.10.12 | System default | Yes |
| Node.js | 20.19.6 | NodeSource | Yes |
| TypeScript | 5.9.3 | npm global | Yes |
| ts-node | 10.9.2 | npm global | Yes |
| Go | 1.18.1 | apt (1.22+ recommended) | Yes |
| Java | 17.0.17 | openjdk-17-jdk | Yes |
| Ruby | 3.0.2 | apt | Yes |
| Rust | 1.92.0 | rustup | Yes |
| **Cargo** | **1.92.0** | **rustup** | **⚠️ CRITICAL** |

> **Note:** Go 1.18+ works but 1.22+ is recommended. If you encounter Go validation errors, consider upgrading Go manually (see Go section above).

## Complete Installation Script

Copy and run this entire block for a complete setup:

```bash
#!/bin/bash
set -e

echo "=== MBPP Code Translations - Complete Setup ==="

# Update system
sudo apt update && sudo apt upgrade -y

# Node.js
echo "Installing Node.js..."
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# TypeScript
echo "Installing TypeScript..."
sudo npm install -g typescript ts-node

# Go
echo "Installing Go..."
sudo apt install -y golang-go

# Java
echo "Installing Java..."
sudo apt install -y openjdk-17-jdk

# Ruby
echo "Installing Ruby..."
sudo apt install -y ruby ruby-dev

# Rust (via rustup for Cargo support)
echo "Installing Rust via rustup..."
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Verify all installations
echo ""
echo "=== Verification ==="
echo "Node.js:    $(node --version)"
echo "TypeScript: $(tsc --version)"
echo "Go:         $(go version)"
echo "Java:       $(java --version 2>&1 | head -1)"
echo "Ruby:       $(ruby --version)"
echo "Rust:       $(rustc --version)"
echo "Cargo:      $(cargo --version)"

echo ""
echo "✅ All runtimes installed!"
echo ""
echo "Next steps:"
echo "1. cd /mnt/c/path/to/MuSR/mbpp-code-translations"
echo "2. pip install -r requirements.txt"
echo "3. export ANTHROPIC_API_KEY='your-key'"
echo "4. python3 translate_mbpp.py --language rust --num-samples 1"
```
