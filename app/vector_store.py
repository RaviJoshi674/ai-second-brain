import requests
import msgpack
import json
from typing import List, Dict, Any, Optional

class EndeeClient:
    """Client for interacting with the local Endee vector database."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        self.base_url = base_url.rstrip('/')
        # Assuming open mode for now as per instructions (no auth needed by default)
        self.headers = {"Content-Type": "application/json"}

    def _get_url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def check_health(self) -> bool:
        """Check if Endee is running."""
        try:
            response = requests.get(self._get_url("/api/v1/health"), timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def create_index(self, index_name: str, dim: int, space_type: str = "cosine") -> bool:
        """Create a new vector index in Endee.
        space_type can be: cosine, l2, ip
        """
        payload = {
            "index_name": index_name,
            "dim": dim,
            "space_type": space_type,
            "M": 16, # Default HNSW M parameter
            "ef_con": 100 # Default HNSW ef_construction parameter
        }
        
        # Check if index already exists. Endee API might return 409 Conflict if it does.
        response = requests.post(
            self._get_url("/api/v1/index/create"),
            headers=self.headers,
            json=payload
        )
        # 200 OK or 409 Conflict (already exists)
        return response.status_code in [200, 409]
        
    def get_index_info(self, index_name: str) -> Optional[Dict[str, Any]]:
        """Get information about an index to test if it exists."""
        response = requests.get(
            self._get_url(f"/api/v1/index/{index_name}/info"),
        )
        if response.status_code == 200:
            return response.json()
        return None

    def insert_vectors(self, index_name: str, vectors: List[Dict[str, Any]]) -> bool:
        """
        Insert multiple vectors into an index.
        vectors should be a list of dicts: 
        [{"id": "id1", "vector": [1.0, 2.0,...], "meta": "{'text':'...'}"}]
        """
        if not vectors:
            return True
            
        # Format payload to match Endee's required structure and meta must be string
        payload = []
        for v in vectors:
            item = {
                "id": str(v["id"]),
                "vector": v["vector"]
            }
            if "meta" in v:
                item["meta"] = json.dumps(v["meta"]) if isinstance(v["meta"], dict) else str(v["meta"])
            payload.append(item)

        response = requests.post(
            self._get_url(f"/api/v1/index/{index_name}/vector/insert"),
            headers=self.headers,
            json=payload
        )
        return response.status_code == 200

    def search(self, index_name: str, query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for top_k most similar vectors.
        Endee returns search results in msgpack format.
        """
        payload = {
            "vector": query_vector,
            "k": k,
            "include_vectors": False
        }

        response = requests.post(
            self._get_url(f"/api/v1/index/{index_name}/search"),
            headers=self.headers,
            json=payload
        )

        if response.status_code != 200:
            print(f"Search failed: {response.text}")
            return []

        # Parse the messagepack response
        try:
            # Endee returns a list of results (ResultSet structure mapped to msgpack)
            # The structure in python msgpack might yield list of tuples/lists depending on how C++ struct was packed
            # Typical msgpack of a vector of structs:
            # [[id1, dist1, id_num1], [id2, dist2, id_num2], ...]
            # or packed objects.
            raw_results = msgpack.unpackb(response.content, raw=False)
            
            results = []
            if raw_results and hasattr(raw_results, '__iter__'):
                 # Inspecting the exact layout returned by Endee (from main.cpp SearchResponse)
                 # We expect each hit to have an id string, distance, and optionally metadata vector
                 # msgpack unpacks it as a list of lists/dicts typically based on struct. 
                 # Let's iterate and safely extract. We'll need to fetch the document's metadata
                 # if not returned, or parse it if it is.
                 for hit in raw_results:
                     # Endee's SearchResult typically packs as: [internal_id, distance, id_str, meta_bytes, vector] (or similar)
                     # Based on the typical msgpack-c behavior of packing structs.
                     # However, to be safe and robust, it's common practice to do a getVector call for metadata if search doesn't return meta cleanly.
                     # For the sake of this test, we will fetch the full vector using getVector API.
                     # We can also attempt to read metadata if it's packed in the hit.
                     if isinstance(hit, list) and len(hit) >= 3:
                          dist = hit[0]
                          doc_id = hit[1]
                          meta = hit[2]
                          results.append({"id": doc_id, "distance": dist, "meta": meta})
                     elif isinstance(hit, dict):
                          doc_id = hit.get("id") or hit.get("external_id")
                          dist = hit.get("distance")
                          meta = hit.get("meta", b"")
                          results.append({"id": doc_id, "distance": dist, "meta": meta})
            
            # For each result, fetch the actual metadata using the Get API
            final_results = []
            for r in results:
                doc_id = r["id"]
                if isinstance(doc_id, bytes):
                    doc_id = doc_id.decode('utf-8', errors='ignore')
                else:
                    doc_id = str(doc_id)
                # Decode meta if it exists and is bytes
                meta_str = ""
                if "meta" in r and isinstance(r["meta"], bytes) and len(r["meta"]) > 0:
                    meta_str = r["meta"].decode('utf-8')
                else:    
                    vector_data = self.get_vector(index_name, doc_id)
                    if vector_data and "meta" in vector_data:
                        meta_str = vector_data.get("meta", "")
                        
                meta_dict = {}
                try:
                    meta_dict = json.loads(meta_str) if meta_str else {}
                except:
                    meta_dict = {"raw": meta_str}
                    
                dist_val = 0.0
                try:
                    dist_val = float(r.get("distance", 0.0))
                except:
                    pass
                    
                final_results.append({
                    "id": doc_id,
                    "distance": dist_val,
                    "metadata": meta_dict
                })
                
            return final_results
            
        except Exception as e:
            import traceback
            print(f"Error parsing msgpack: {e}")
            traceback.print_exc()
            return []

    def get_vector(self, index_name: str, vector_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a specific vector by ID to get its metadata."""
        payload = {"id": vector_id}
        response = requests.post(
            self._get_url(f"/api/v1/index/{index_name}/vector/get"),
            headers=self.headers,
            json=payload
        )
        if response.status_code == 200:
            try:
                # Returned as msgpack
                raw_vector = msgpack.unpackb(response.content, raw=False)
                # Parse hybrid vector object structure
                vec_obj = {}
                if isinstance(raw_vector, list) and len(raw_vector) >= 3:
                    vid = raw_vector[0]
                    vec_obj["id"] = vid.decode('utf-8') if isinstance(vid, bytes) else str(vid)
                    vec_meta = raw_vector[1]
                    vec_obj["meta"] = vec_meta.decode('utf-8') if isinstance(vec_meta, bytes) else str(vec_meta)
                elif isinstance(raw_vector, dict):
                    vec_obj = raw_vector
                    if "meta" in vec_obj and isinstance(vec_obj["meta"], bytes):
                        vec_obj["meta"] = vec_obj["meta"].decode('utf-8')
                return vec_obj
            except Exception as e:
                print(f"Error parsing get_vector msgpack: {e}")
                pass
        return None
