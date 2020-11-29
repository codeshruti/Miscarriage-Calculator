mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" 
const PORT = process.env.PORT || '8080' 
app=express();
app.set("port",PORT);> ~/ .streamlit/config.toml
