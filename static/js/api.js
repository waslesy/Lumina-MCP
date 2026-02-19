const API_KEY_STORAGE_KEY = 'lumina_master_key';

function getMasterKey() {
    let key = localStorage.getItem(API_KEY_STORAGE_KEY);
    // 简单的交互：如果本地没有 key，就弹窗让用户输一次（MVP 方案）
    if (!key) {
        key = prompt("请输入 Lumina Master Key (管理员密钥):");
        if (key) {
            localStorage.setItem(API_KEY_STORAGE_KEY, key);
        } else {
            throw new Error("未提供密钥，无法访问");
        }
    }
    return key;
}

const API = {
    async request(method, url, data = null) {
        const headers = {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${getMasterKey()}` // 注入 Master Key
        };

        const config = { method, headers };
        if (data) config.body = JSON.stringify(data);

        const res = await fetch(url, config);
        
        // 如果 401 未授权，可能是 key 错了，清除本地存储重试
        if (res.status === 401) {
            localStorage.removeItem(API_KEY_STORAGE_KEY);
            alert("密钥错误或过期，请刷新页面重新输入。");
            throw new Error("Unauthorized");
        }

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || '请求失败');
        }
        return res.json();
    },

    get(url) { return this.request('GET', url); },
    post(url, data) { return this.request('POST', url, data); },
    put(url, data) { return this.request('PUT', url, data); },
    delete(url) { return this.request('DELETE', url); }
};