"""Boundary checks for the model-artifact and credential publication guard."""
import unittest
from check_staged import violations, MAX_BYTES

class PublicationGuardTests(unittest.TestCase):
    def test_weights_and_optimizer_state_are_rejected(self):
        for name in ['model.safetensors','optimizer.pt','adapter.bin','weights.gguf','x/checkpoint-10/config.json']:
            self.assertIn('model-or-array-artifact', violations(name,b''))

    def test_ordinary_configuration_and_secret_file_reference_are_allowed(self):
        self.assertEqual(violations('configs/train.json',b'{"bf16":true,"api_key":"EMPTY"}'),[])
        self.assertEqual(violations('scripts/auth.py',b'path="/home/user/.secrets/key"'),[])

    def test_secret_formats_are_detected_without_printing_values(self):
        self.assertIn('github-token',violations('config.py',('ghp_'+'a'*36).encode()))
        self.assertIn('jwt',violations('config.json',('eyJ'+'a'*20+'.'+'b'*20+'.'+'c'*30).encode()))

    def test_oversized_files_and_nested_repositories_are_rejected(self):
        self.assertIn('over-5-MiB',violations('large.json',b'',MAX_BYTES+1))
        self.assertIn('nested-git-repository',violations('upstream',b'',mode='160000'))

    def test_external_symlinks_are_rejected(self):
        self.assertIn('external-symlink',violations('weights',b'/scratch/model',mode='120000'))
        self.assertIn('external-symlink',violations('weights',b'../../model',mode='120000'))

if __name__=='__main__':
    unittest.main()
