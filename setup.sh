#!/bin/bash

VENV_DIR=".venv"
DATA_DIR="data"
REQUIREMENTS="requirements.txt"
DATASETS_FILE="datasets.txt"
TEMP_DIR="temp_extract"

clean_dataset() {
  local input="$1"
  local output="$2"
  local name="$3"

  case "$name" in
    "adult")
      echo "age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,income" > "$output"
      cat "$input" >> "$output"
      ;;
    "banknote")
      echo "variance,skewness,curtosis,entropy,class" > "$output"
      cat "$input" >> "$output"
      ;;
    "heart")
      echo "age sex cp trestbps chol fbs restecg thalach exang oldpeak slope ca thal target" | tr ' ' ',' > "$output"
      tr -s '[:blank:]' ',' < "$input" >> "$output"
      ;;
    "iris")
      echo "sepal_length,sepal_width,petal_length,petal_width,class" > "$output"
      cat "$input" >> "$output"
      ;;
    "seeds")
      echo "area,perimeter,compactness,kernel_length,kernel_width,asymmetry,groove_length,class" > "$output"
      tr '\t' ',' < "$input" >> "$output"
      ;;
    "wine")
      # Optional: convert ; to , for compatibility
      tr ';' ',' < "$input" > "$output"
      ;;
    *)
      echo "Unknown dataset format for $name. Copying raw file."
      cp "$input" "$output"
      ;;
  esac
}


# Detect platform to use correct activate path
IS_WINDOWS=false
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) IS_WINDOWS=true ;;
esac

# Create virtual environment if not present
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
  echo "✅ Virtual environment created."
fi

# Activate virtual environment
if [ "$IS_WINDOWS" = true ]; then
  # Windows (Git Bash): use Scripts
  source "$VENV_DIR/Scripts/activate"
else
  # Unix-based systems
  source "$VENV_DIR/bin/activate"
fi

echo "✅ Virtual environment activated."

# Install requirements
if [ -f "$REQUIREMENTS" ]; then
  pip install --quiet -r "$REQUIREMENTS"
  echo "✅ Requirements installed."
fi

# Create data directory
mkdir -p "$DATA_DIR"

# Download datasets
if [ -f "$DATASETS_FILE" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue

    url=$(echo "$line" | awk '{print $1}')
    zipname=$(echo "$line" | awk '{print $2}' | tr -d '\r')
    base="${zipname%.zip}"
    target_csv="$DATA_DIR/$base.csv"

    if [ -f "$target_csv" ]; then
      echo "✅ $base.csv already exists. Skipping."
      continue
    fi

    echo "📥 Downloading $zipname..."
    curl -sL "$url" -o "$zipname"

    echo "📦 Extracting $zipname..."
    mkdir -p "$TEMP_DIR"
    unzip -qq "$zipname" -d "$TEMP_DIR"

    # Try to find a likely data file
    data_file=$(find "$TEMP_DIR" -type f \( -iname "*.data" -o -iname "*.txt" -o -iname "*.csv" \) | head -n 1)

    if [ -z "$data_file" ]; then
      echo "❌ No data file found in $zipname. Skipping."
    else
      echo "📄 Found data file: $data_file"
      clean_dataset "$data_file" "$target_csv" "$base"
      echo "✅ Saved as $target_csv"
    fi

    # Clean up
    rm "$zipname"
    rm -rf "$TEMP_DIR"

  done < "$DATASETS_FILE"
else
  echo "⚠️ $DATASETS_FILE not found."
fi
