import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import hashlib
import numpy as np
from collections import defaultdict
import re


@dataclass
class Message:
    role: str  
    content: str
    timestamp: str
    context_summary: Optional[str] = None
    embeddings: Optional[List[float]] = None
    topic_tags: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def get_preview(self, max_length: int = 100) -> str:
        if len(self.content) > max_length:
            return self.content[:max_length] + "..."
        return self.content


class Level1ConversationMemory:
    def __init__(self, max_messages: int = 3):
        self.max_messages = max_messages
        self.messages: List[Message] = []
        self.summaries: List[str] = []
    
    def add_message(self, role: str, content: str, topic_tags: Optional[List[str]] = None) -> None:
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            topic_tags=topic_tags or []
        )
        
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            removed = self.messages.pop(0)
            self._create_summary_for_old_message(removed)
    
    def _create_summary_for_old_message(self, message: Message) -> None:
        summary = f"[{message.role.upper()}] {message.get_preview(50)}"
        self.summaries.append(summary)
        if len(self.summaries) > 5:
            self.summaries.pop(0)
    
    def get_context_summary(self) -> str:
        if not self.summaries:
            return "No previous context"
        return "\n".join(self.summaries)
    
    def get_recent_context(self) -> str:
        if not self.messages:
            return "No conversation history"
        
        context_parts = []
        for msg in self.messages:
            context_parts.append(f"{msg.role.upper()}: {msg.get_preview(80)}")
        
        return "\n".join(context_parts)
    
    def build_prompt_with_context(self, new_query: str) -> Tuple[str, str]:
        previous_context = self.get_context_summary()
        recent_context = self.get_recent_context()
        
        prompt = f"""
## Previous Conversation Summary
{previous_context}

## Recent Conversation Context
{recent_context}

## New Query
{new_query}

Please answer considering the full conversation context above.
"""
        
        return prompt, f"{previous_context}\n{recent_context}"
    
    def get_all_messages(self) -> List[Dict]:
        """Get all messages as dict"""
        return [msg.to_dict() for msg in self.messages]
    
    def clear(self) -> None:
        """Clear all messages"""
        self.messages.clear()
        self.summaries.clear()

class TopicTracker:
    
    def __init__(self, db_path: str = "conversation_topics.db"):
        self.db_path = db_path
        self.current_topics: Dict[str, List[str]] = defaultdict(list)
        
        try:
            self.init_database()
        except Exception as e:
            print(f"⚠️ Database init error: {e}. Attempting recovery...")
            import os
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.init_database()
    
    def init_database(self) -> None:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY,
                topic_name TEXT UNIQUE,
                created_at TIMESTAMP,
                last_updated TIMESTAMP,
                message_count INTEGER DEFAULT 0,
                topic_hash TEXT
            )
        ''')
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS topic_messages (
                id INTEGER PRIMARY KEY,
                topic_id INTEGER,
                message_content TEXT,
                message_role TEXT,
                timestamp TIMESTAMP,
                similarity_score REAL,
                FOREIGN KEY (topic_id) REFERENCES topics(id)
            )
        ''')
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS topic_keywords (
                id INTEGER PRIMARY KEY,
                topic_id INTEGER,
                keyword TEXT,
                frequency INTEGER,
                FOREIGN KEY (topic_id) REFERENCES topics(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def extract_keywords(self, text: str) -> List[str]:
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        word_freq = defaultdict(int)
        
        for word in words:
            word_freq[word] += 1
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        return [kw[0] for kw in keywords]
    
    def compute_text_similarity(self, text1: str, text2: str) -> float:
        keywords1 = set(self.extract_keywords(text1))
        keywords2 = set(self.extract_keywords(text2))
        
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = len(keywords1.intersection(keywords2))
        union = len(keywords1.union(keywords2))
        return intersection / union if union > 0 else 0.0
    
    def detect_or_create_topic(self, message: str, embeddings_model=None) -> Tuple[str, float]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('SELECT id, topic_name FROM topics ORDER BY last_updated DESC')
        existing_topics = c.fetchall()
        
        best_match = None
        best_score = 0.0

        for topic_id, topic_name in existing_topics[-10:]: 
            c.execute('''
                SELECT message_content FROM topic_messages 
                WHERE topic_id = ? 
                ORDER BY timestamp DESC LIMIT 1
            ''', (topic_id,))
            
            result = c.fetchone()
            if result:
                similarity = self.compute_text_similarity(message, result[0])
                if similarity > best_score:
                    best_score = similarity
                    best_match = (topic_id, topic_name)

        if best_score > 0.65 and best_match:
            self._add_message_to_topic(best_match[0], message, best_score, conn)
            conn.close()
            return best_match[1], best_score
        new_topic_name = self._generate_topic_name(message)
        self._create_new_topic(new_topic_name, message, conn)
        conn.close()
        
        return new_topic_name, 1.0
    
    def _generate_topic_name(self, text: str) -> str:
        words = re.findall(r'\b[A-Z][a-z]+\b', text[:200])
        
        if words:
            topic = " ".join(words[:3])
        else:
            topic = "Topic_" + hashlib.md5(text.encode()).hexdigest()[:8]
        
        return topic
    
    def _create_new_topic(self, topic_name: str, first_message: str, conn) -> int:
        c = conn.cursor()
        
        topic_hash = hashlib.md5(topic_name.encode()).hexdigest()
        timestamp = datetime.now()
        
        try:
            c.execute('''
                INSERT INTO topics (topic_name, created_at, last_updated, message_count, topic_hash)
                VALUES (?, ?, ?, 1, ?)
            ''', (topic_name, timestamp, timestamp, topic_hash))
            
            topic_id = c.lastrowid
            c.execute('''
                INSERT INTO topic_messages (topic_id, message_content, message_role, timestamp, similarity_score)
                VALUES (?, ?, 'user', ?, 1.0)
            ''', (topic_id, first_message, timestamp))

            keywords = self.extract_keywords(first_message)
            for keyword in keywords:
                c.execute('''
                    INSERT INTO topic_keywords (topic_id, keyword, frequency)
                    VALUES (?, ?, 1)
                ''', (topic_id, keyword))
            
            conn.commit()
            return topic_id
        
        except sqlite3.IntegrityError:
            c.execute('SELECT id FROM topics WHERE topic_name = ?', (topic_name,))
            return c.fetchone()[0]
    
    def _add_message_to_topic(self, topic_id: int, message: str, similarity: float, conn) -> None:
        c = conn.cursor()
        timestamp = datetime.now()
        
        c.execute('''
            INSERT INTO topic_messages (topic_id, message_content, message_role, timestamp, similarity_score)
            VALUES (?, ?, 'user', ?, ?)
        ''', (topic_id, message, timestamp, similarity))
        
        c.execute('''
            UPDATE topics SET last_updated = ?, message_count = message_count + 1
            WHERE id = ?
        ''', (timestamp, topic_id))
        
        conn.commit()
    
    def get_all_topics(self) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('SELECT id, topic_name, created_at, message_count FROM topics ORDER BY last_updated DESC')
        topics = []
        
        for topic_id, name, created, count in c.fetchall():
            c.execute('''
                SELECT message_content FROM topic_messages WHERE topic_id = ? LIMIT 3
            ''', (topic_id,))
            
            messages = [row[0] for row in c.fetchall()]
            
            topics.append({
                'id': topic_id,
                'name': name,
                'created': created,
                'message_count': count,
                'recent_messages': messages
            })
        
        conn.close()
        return topics
    
    def get_topic_by_name(self, topic_name: str) -> Optional[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('SELECT id, created_at, message_count FROM topics WHERE topic_name = ?', (topic_name,))
        result = c.fetchone()
        
        if not result:
            conn.close()
            return None
        
        topic_id, created, count = result
        
        c.execute('''
            SELECT message_content, message_role, similarity_score FROM topic_messages 
            WHERE topic_id = ? ORDER BY timestamp DESC
        ''', (topic_id,))
        
        messages = [
            {'content': row[0], 'role': row[1], 'similarity': row[2]}
            for row in c.fetchall()
        ]
        
        c.execute('''
            SELECT keyword, frequency FROM topic_keywords WHERE topic_id = ?
            ORDER BY frequency DESC LIMIT 10
        ''', (topic_id,))
        
        keywords = [row[0] for row in c.fetchall()]
        
        conn.close()
        
        return {
            'id': topic_id,
            'name': topic_name,
            'created': created,
            'message_count': count,
            'messages': messages,
            'keywords': keywords
        }


class ConversationUpdater:
    def __init__(self, topic_tracker: TopicTracker):
        self.topic_tracker = topic_tracker
        self.conversation_branches: Dict[str, List[Dict]] = defaultdict(list)
        self.update_history: List[Dict] = []
    
    def append_to_conversation(self, topic_name: str, new_message: str, message_type: str = "update") -> Dict[str, Any]:
        result = {
            'action': 'append',
            'topic': topic_name,
            'timestamp': datetime.now().isoformat(),
            'message': new_message,
            'type': message_type
        }

        conn = sqlite3.connect(self.topic_tracker.db_path)
        c = conn.cursor()
        
        c.execute('SELECT id FROM topics WHERE topic_name = ?', (topic_name,))
        topic_result = c.fetchone()
        
        if topic_result:
            topic_id = topic_result[0]
            self.topic_tracker._add_message_to_topic(topic_id, new_message, 0.9, conn)
            result['status'] = 'added_to_existing_topic'
        else:
            result['status'] = 'topic_not_found'
        
        conn.close()
        self.update_history.append(result)
        return result
    
    def branch_conversation(self, topic_name: str, branch_name: str, branch_content: str) -> Dict[str, Any]:
        branch_id = f"{topic_name}_{branch_name}_{datetime.now().timestamp()}"
        
        branch = {
            'branch_id': branch_id,
            'parent_topic': topic_name,
            'branch_name': branch_name,
            'content': branch_content,
            'created_at': datetime.now().isoformat(),
            'children': []
        }
        self.conversation_branches[topic_name].append(branch)
        try:
            conn = sqlite3.connect(self.topic_tracker.db_path)
            c = conn.cursor()
            c.execute('SELECT id FROM topics WHERE topic_name = ?', (topic_name,))
            topic_result = c.fetchone()
            
            if topic_result:
                topic_id = topic_result[0]
                branch_message = f"[BRANCH:{branch_name}] {branch_content}"
                c.execute('''
                    INSERT INTO topic_messages (topic_id, message_content, message_role, timestamp, similarity_score)
                    VALUES (?, ?, 'branch', ?, 1.0)
                ''', (topic_id, branch_message, datetime.now()))
                
                conn.commit()
        except Exception as e:
            print(f"⚠️ Could not persist branch: {e}")
        finally:
            if conn:
                conn.close()
        
        result = {
            'action': 'branch',
            'topic': topic_name,
            'branch_id': branch_id,
            'branch_name': branch_name,
            'status': 'created',
            'persisted': True
        }
        
        self.update_history.append(result)
        return result
    
    def merge_branches(self, parent_topic: str, branch_ids: List[str]) -> Dict[str, Any]:
        merged_content = []
        
        for branch_id in branch_ids:
            for branch in self.conversation_branches[parent_topic]:
                if branch['branch_id'] == branch_id:
                    merged_content.append(branch['content'])
                    break
        
        merged_text = "\n---\n".join(merged_content)
        
        result = {
            'action': 'merge',
            'topic': parent_topic,
            'merged_branches': branch_ids,
            'merged_content': merged_text,
            'timestamp': datetime.now().isoformat(),
            'status': 'merged'
        }
        
        self.update_history.append(result)
        return result
    
    def get_conversation_evolution(self, topic_name: str) -> Dict[str, Any]:
        topic = self.topic_tracker.get_topic_by_name(topic_name)
        
        if not topic:
            return {'error': 'Topic not found'}
        branches = []
        if topic.get('messages'):
            for msg in topic['messages']:
                if msg['role'] == 'branch' and msg['content'].startswith('[BRANCH:'):
                    try:
                        parts = msg['content'].split('] ', 1)
                        branch_name = parts[0].replace('[BRANCH:', '')
                        branch_content = parts[1] if len(parts) > 1 else ''
                        
                        branches.append({
                            'branch_id': f"{topic_name}_{branch_name}",
                            'parent_topic': topic_name,
                            'branch_name': branch_name,
                            'content': branch_content,
                            'created_at': msg.get('timestamp', 'unknown'),
                            'children': []
                        })
                    except:
                        pass
        
        memory_branches = self.conversation_branches.get(topic_name, [])
        for branch in memory_branches:
            if not any(b['branch_name'] == branch['branch_name'] for b in branches):
                branches.append(branch)
        
        return {
            'topic': topic_name,
            'main_thread': topic['messages'],
            'keywords': topic['keywords'],
            'created': topic['created'],
            'total_messages': topic['message_count'],
            'branches': branches,
            'branch_count': len(branches)
        }
    
    def get_update_timeline(self) -> List[Dict]:
        for update in self.update_history:
            if 'timestamp' not in update:
                update['timestamp'] = datetime.now().isoformat()
        
        return sorted(self.update_history, key=lambda x: x.get('timestamp', ''), reverse=True)

class ConversationContinuitySystem:
    def __init__(self):
        self.level1_memory = Level1ConversationMemory(max_messages=3)
        self.level2_tracker = TopicTracker()
        self.level3_updater = ConversationUpdater(self.level2_tracker)
    
    def process_message(self, message: str, role: str = "user", message_type: str = "query") -> Dict[str, Any]:
        self.level1_memory.add_message(role, message)

        topic_name, similarity = self.level2_tracker.detect_or_create_topic(message)
        evolution = self.level3_updater.get_conversation_evolution(topic_name)
        
        return {
            'level1_context': self.level1_memory.get_recent_context(),
            'level2_topic': topic_name,
            'level2_similarity': similarity,
            'level3_evolution': evolution,
            'timestamp': datetime.now().isoformat(),
            'message_type': message_type
        }
    
    def get_full_status(self) -> Dict[str, Any]:
        return {
            'level1': {
                'recent_messages': self.level1_memory.get_all_messages(),
                'context_summary': self.level1_memory.get_context_summary()
            },
            'level2': {
                'total_topics': len(self.level2_tracker.get_all_topics()),
                'topics': self.level2_tracker.get_all_topics()
            },
            'level3': {
                'total_branches': sum(len(v) for v in self.level3_updater.conversation_branches.values()),
                'update_timeline': self.level3_updater.get_update_timeline()
            }
        }

