<!-- srcbook:{"language":"typescript","tsconfig.json":{"compilerOptions":{"module":"nodenext","moduleResolution":"nodenext","target":"es2022","resolveJsonModule":true,"noEmit":true,"allowImportingTsExtensions":true},"include":["src/**/*"],"exclude":["node_modules"]}} -->

# sparkastML intention classification

###### package.json

```json
{
  "type": "module",
  "dependencies": {
    "@types/node": "latest",
    "@xenova/transformers": "^2.17.2",
    "onnxruntime-web": "^1.19.0",
    "tsx": "latest",
    "typescript": "latest"
  }
}

```

###### tokenizer.ts

```typescript
export type TokenDict = { [key: string]: number };

type TokenDict = { [key: string]: number };

function tokenize(query: string, tokenDict: TokenDict): number[] {
    const tokenIds: number[] = [];
    let index = 0;

    // Replace spaces with "▁"
    query = "▁"+query.replace(/ /g, "▁");
    query = query.replace(/\n/g, "<0x0A>");

     while (index < query.length) {
        let bestToken = null;
        let bestLength = 0;

        // Step 2: Find the longest token that matches the beginning of the remaining query
        for (const token in tokenDict) {
            if (query.startsWith(token, index) && token.length > bestLength) {
                bestToken = token;
                bestLength = token.length;
            }
        }

        if (bestToken) {
            tokenIds.push(tokenDict[bestToken]);
            index += bestLength;
        } else {
            // Step 3: Handle the case where no token matches
            const char = query[index];
            if (char.charCodeAt(0) <= 127) {
                // If the character is ASCII, and it doesn't match any token, treat it as an unknown token
                throw new Error(`Unknown token: ${char}`);
            } else {
                // If the character is non-ASCII, convert it to a series of bytes and match each byte
                const bytes = new TextEncoder().encode(char);
                for (const byte of bytes) {
                    const byteToken = `<0x${byte.toString(16).toUpperCase()}>`;
                    if (tokenDict[byteToken] !== undefined) {
                        tokenIds.push(tokenDict[byteToken]);
                    } else {
                        throw new Error(`Unknown byte token: ${byteToken}`);
                    }
                }
            }
            index += 1;
        }
    }

    return tokenIds;
}

export default tokenize
```

###### embedding.ts

```typescript
import * as fs from 'fs';
import * as path from 'path';

type EmbeddingDict = { [key: number]: Float32Array };

function getEmbeddingLayer(buffer: Buffer): EmbeddingDict {
    const dict: EmbeddingDict = {};

    const entrySize = 514;
    const numEntries = buffer.length / entrySize;

    for (let i = 0; i < numEntries; i++) {
        const offset = i * entrySize;
        const key = buffer.readUInt16LE(offset);
        const floatArray = new Float32Array(128);

        for (let j = 0; j < 128; j++) {
            floatArray[j] = buffer.readFloatLE(offset + 2 + j * 4);
        }

        dict[key] = floatArray;
    }

    return dict;
}

function getEmbedding(tokenIds: number[], embeddingDict: EmbeddingDict, contextSize: number) {
  let result = [];
  for (let i = 0; i < contextSize; i++) {
    if (i < tokenIds.length) {
      const tokenId = tokenIds[i];
      result = result.concat(Array.from(embeddingDict[tokenId]))
    }
    else {
      result = result.concat(new Array(128).fill(0))
    }
  }
  return new Float32Array(result);
}

export {getEmbeddingLayer, getEmbedding};
```

###### load.ts

```typescript
import * as ort from 'onnxruntime-web';
import * as fs from 'fs';
import tokenize, {TokenDict} from "./tokenizer.ts"
import {getEmbeddingLayer, getEmbedding} from "./embedding.ts"

const embedding_file = './token_embeddings.bin';
const embedding_data = fs.readFileSync(embedding_file);
const embedding_buffer = Buffer.from(embedding_data);
const query = `Will it rain tomorrow`;
const model_path = './model.onnx';
const vocabData = fs.readFileSync('./token_to_id.json');
const vocabDict = JSON.parse(vocabData.toString());

let lastLogCall = new Date().getTime();

function log(task: string) {
  const currentTime = new Date().getTime();
  const costTime = currentTime - lastLogCall;
  console.log(`[${currentTime}] (+${costTime}ms) ${task}`)
  lastLogCall = new Date().getTime();
}

async function loadModel(modelPath: string) {
  const session = await ort.InferenceSession.create(modelPath);
  return session;
}

async function runInference(query: string, embedding_buffer: Buffer, modelPath: string, vocabDict: TokenDict) {
  const session = await loadModel(modelPath);
  log("loadModel:end");
  const inputText = query;
  const queryLength = query.length;
  const tokenIds = await tokenize(query, vocabDict);
  log("tokenize:end");
  const embeddingDict = getEmbeddingLayer(embedding_buffer);
  const e = getEmbedding(tokenIds, embeddingDict, 12);
  log("getEmbedding:end");

  const inputTensor = new ort.Tensor('float32', e, [1, 12, 128]);

  const feeds = { 'input': inputTensor };
  const results = await session.run(feeds);
  log("inference:end");

  const output = results.output.data;
  const predictedClassIndex = output.indexOf(Math.max(...output));

  return  output;
}

console.log("Perdicted class:", await runInference(query, embedding_buffer, model_path, vocabDict));
```
