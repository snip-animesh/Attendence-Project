class Trie:
    head = {}

    def insert(self, word):
        current_node = self.head

        for ch in word:
            if ch not in current_node:
                current_node[ch] = {}
            current_node = current_node[ch]
        current_node['*'] = True

    def search(self, word):
        current_node = self.head

        for ch in word:
            if ch not in current_node:
                return False
            current_node = current_node[ch]
        if "*" in current_node:
            return True
        else:
            return False

    # def printf(self):
    #     print(self.head)


if __name__ == "__main__":
    trie = Trie()
    trie.insert("Roll")
    trie.insert("1801077")
    trie.insert('1801070')
    print(trie.search('1801069'))
    print(trie.search('1801070'))
